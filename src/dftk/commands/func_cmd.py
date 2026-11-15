"""
dftk.commands.func_cmd — dftk func subcommand.

Applies column transforms, optionally within groups.  The main use case is
adding derived columns to a DataFrame: running totals, quantile bins, group-
level aggregates broadcast back to the original row count, etc.

TRANSFORMS (-t / --transform)
------------------------------
  sum       Group sum broadcast to each row (or simple column sum).
  mean      Group mean broadcast to each row.
  min       Group min broadcast to each row.
  max       Group max broadcast to each row.
  count     Non-null count per group broadcast to each row.
  median    Group median broadcast to each row.
  std       Group standard deviation broadcast to each row.
  cumsum    Cumulative sum (within group when -g is given).
  rank      Rank within group (average method, ascending).
  pct_rank  Percentile rank within group (0–1).
  qcut:N    Assign each value to one of N equal-frequency quantile bins.
            The bin label is an integer 1..N.
  interpolate  Fill empty cells on a straight line, against --xcol X,
            between the known values before and after them (within each
            group with -g), so unevenly spaced rows get the right values.
            Gaps before a group's first value or after its last stay
            empty: nothing is extrapolated.
  ffill     Fill empty cells with the value before them in --xcol order
            (bfill: after), within each group.

FILLING GAPS (interpolate, ffill, bfill)
----------------------------------------
These need -x/--xcol X (time, position, year, date, ...): the rows are
taken in X order within each group, whatever order the file is in, and
written back in the file's order, so no sort step is needed.  Rows without
an X value are left as they are.  Only empty cells are filled: a missing
row (a year with no row at all) is not added.

interpolate needs a numeric column, and a numeric or date X; its result is
always decimal (an integer column comes back as 1.0, 2.0, ...).  Repeated X
values within a group are an error (which value would the line go
through?), even in a group with nothing to fill: average replicates first
(pivot) or add a replicate column to -g.  ffill and bfill take any column
and any X that can be sorted (numbers, dates, text); rows with the same X
keep their file order.

For no gaps at all, chain all three on the same column (-d): interpolate,
then ffill carries the last known value on to the end and bfill the first
known value back to the start.

interpolate handles one group at a time: a million rows in 10,000 groups
take about half a second, but a million groups of one or two rows take
about ten seconds.  ffill and bfill don't depend on the number of groups.

EXAMPLES
--------
Cumulative sum of "value":

  dftk func data.tsv -c value -t cumsum
  dftk dataset tips -o | dftk func -c total_bill -t cumsum

Group mean broadcast (adds "total_bill_mean" unless -d is given):

  dftk dataset tips -o | dftk func -c total_bill -g day -t mean

Quantile bins (quartiles):

  dftk dataset tips -o | dftk func -c tip -t qcut:4 -d tip_quartile

Fill gaps in life expectancy (here every year ending in 5 blanked out) on
a straight line between each country's neighbouring years:

  dftk dataset healthexp -o \\
      | dftk eval -f "le = Life_Expectancy.where(Year % 10 != 5)" -o \\
      | dftk func -c le -t interpolate -g Country --xcol Year -d le_filled

The value before instead (last observation carried forward):

  dftk dataset healthexp -o \\
      | dftk eval -f "le = Life_Expectancy.where(Year % 10 != 5)" -o \\
      | dftk func -c le -t ffill -g Country --xcol Year -d le_filled

No gaps left: interpolate, then carry the end values out to the ends:

  dftk dataset healthexp -o \\
      | dftk eval -f "le = Life_Expectancy.where(Year % 10 != 0)" -o \\
      | dftk func -c le -t interpolate -g Country --xcol Year -d le_filled -o \\
      | dftk func -c le_filled -t ffill -g Country --xcol Year -d le_filled -o \\
      | dftk func -c le_filled -t bfill -g Country --xcol Year -d le_filled

Multiple transforms in one call:

  dftk dataset tips -o | dftk func -c total_bill -t cumsum -d bill_cumsum -o \\
      | dftk func -c total_bill -t mean -g day -d bill_daymean
"""

import argparse
import re

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

# Transforms that use groupby().transform()
_GROUPBY_TRANSFORMS = {"sum", "mean", "min", "max", "count", "median", "std"}

# Transforms that use cumsum / rank within an optional group
_CUMULATIVE_TRANSFORMS = {"cumsum", "rank", "pct_rank"}

# Transforms that fill empty cells, in --xcol order (required)
_FILL_TRANSFORMS = {"interpolate", "ffill", "bfill"}

_VALID = (
    "sum, mean, min, max, count, median, std, cumsum, rank, pct_rank, qcut:N, "
    "interpolate, ffill, bfill"
)


def _group_indices(df: pd.DataFrame, groupcols: list[str] | None):
    """(group name, row positions) for each group, or one group of all rows."""
    if not groupcols:
        return [(None, np.arange(len(df)))]
    gb = df.groupby(groupcols, sort=False, dropna=False)
    return list(gb.indices.items())


def _group_label(name, groupcols) -> str:
    if not isinstance(name, tuple):
        name = (name,)
    return ", ".join(f"{c}={v}" for c, v in zip(groupcols, name, strict=True))


def _interp_x(x: pd.Series, xcol: str) -> np.ndarray:
    """X as floats for np.interp: numbers as they are, dates as nanoseconds."""
    if pd.api.types.is_bool_dtype(x):
        raise ValueError(f"--xcol {xcol!r} is true/false; interpolate needs numbers.")
    if pd.api.types.is_datetime64_any_dtype(x):
        if getattr(x.dt, "tz", None) is not None:
            x = x.dt.tz_convert("UTC").dt.tz_localize(None)
        ns = x.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float)
        return np.where(x.isna().to_numpy(), np.nan, ns)
    if not pd.api.types.is_numeric_dtype(x):
        raise ValueError(
            f"--xcol {xcol!r} must be numbers or dates for interpolate "
            "(ffill and bfill also take text)."
        )
    return x.to_numpy(dtype=float, na_value=np.nan)


def _sort_key(x: pd.Series) -> np.ndarray:
    """Rank of each X value (ties share one), -1 where X is missing."""
    return pd.factorize(x, sort=True, use_na_sentinel=True)[0]


def _fill_gaps(
    df: pd.DataFrame,
    col: str,
    transform: str,
    groupcols: list[str] | None,
    xcol: str,
) -> pd.Series:
    """interpolate / ffill / bfill *col* within groups, in *xcol* order."""
    values = df[col]

    if transform == "interpolate":
        if pd.api.types.is_bool_dtype(values) or not pd.api.types.is_numeric_dtype(
            values
        ):
            raise ValueError(f"interpolate needs a numeric column; {col!r} is not.")
        xall = _interp_x(df[xcol], xcol)
        v = values.to_numpy(dtype=float, na_value=np.nan)
        out = v.copy()
        for name, idx in _group_indices(df, groupcols):
            x = xall[idx]
            has_x = ~np.isnan(x)
            xs = x[has_x]
            if len(np.unique(xs)) < len(xs):
                where = (
                    f" in group {_group_label(name, groupcols)}" if groupcols else ""
                )
                raise ValueError(
                    f"Repeated --xcol {xcol!r} values{where}; interpolate needs "
                    "one row per X value in each group."
                )
            known = has_x & ~np.isnan(v[idx])
            if known.sum() < 2:
                continue
            kx, kv = x[known], v[idx][known]
            order = np.argsort(kx)
            kx, kv = kx[order], kv[order]
            gap = has_x & np.isnan(v[idx]) & (x > kx[0]) & (x < kx[-1])
            out[idx[gap]] = np.interp(x[gap], kx, kv)
        return pd.Series(out, index=df.index, name=col)

    # ffill / bfill: lexsort is stable, so rows with the same X keep file order
    key = _sort_key(df[xcol])
    codes = (
        df.groupby(groupcols, sort=False, dropna=False).ngroup().to_numpy()
        if groupcols
        else np.zeros(len(df), dtype=np.int64)
    )
    rows = np.flatnonzero(key >= 0)
    order = rows[np.lexsort((key[rows], codes[rows]))]
    part = values.iloc[order].reset_index(drop=True)
    grouped = part.groupby(codes[order])
    filled = grouped.ffill() if transform == "ffill" else grouped.bfill()
    out = values.copy()
    out.iloc[order] = filled.to_numpy()
    return out


def _apply_transform(
    df: pd.DataFrame,
    col: str,
    transform: str,
    groupcols: list[str] | None,
    destcol: str,
    xcol: str | None = None,
) -> pd.DataFrame:
    """Apply *transform* on *col* within optional *groupcols*, storing result in *destcol*."""  # noqa: E501

    if transform in _FILL_TRANSFORMS:
        if xcol is None:
            raise ValueError(
                f"{transform} needs -x/--xcol: the column that orders each "
                "group's rows (time, position, year, ...)."
            )
        df[destcol] = _fill_gaps(df, col, transform, groupcols, xcol)
        return df
    if xcol is not None:
        raise ValueError("--xcol is only for interpolate, ffill and bfill.")

    # --- qcut:N ---
    m = re.fullmatch(r"qcut:(\d+)", transform)
    if m:
        n = int(m.group(1))
        if n < 2:
            raise ValueError("qcut requires N >= 2")
        if groupcols:

            def _qcut_group(s: pd.Series) -> pd.Series:
                return (
                    pd.qcut(s, n, labels=False, duplicates="drop").astype("Int64") + 1
                )  # noqa: E501

            df[destcol] = df.groupby(groupcols)[col].transform(_qcut_group)
        else:
            df[destcol] = (
                pd.qcut(df[col], n, labels=False, duplicates="drop").astype("Int64") + 1
            )
        return df

    # --- groupby().transform() aggregates ---
    if transform in _GROUPBY_TRANSFORMS:
        if groupcols:
            df[destcol] = df.groupby(groupcols)[col].transform(transform)
        else:
            agg_val = getattr(df[col], transform)()
            df[destcol] = agg_val
        return df

    # --- cumsum ---
    if transform == "cumsum":
        if groupcols:
            df[destcol] = df.groupby(groupcols)[col].transform("cumsum")
        else:
            df[destcol] = df[col].cumsum()
        return df

    # --- rank ---
    if transform == "rank":
        if groupcols:
            df[destcol] = df.groupby(groupcols)[col].rank(method="average")
        else:
            df[destcol] = df[col].rank(method="average")
        return df

    # --- pct_rank ---
    if transform == "pct_rank":
        if groupcols:
            df[destcol] = df.groupby(groupcols)[col].rank(method="average", pct=True)
        else:
            df[destcol] = df[col].rank(method="average", pct=True)
        return df

    raise ValueError(f"Unknown transform {transform!r}.  Valid: {_VALID}")


def _default_destcol(col: str, transform: str) -> str:
    """Generate a destination column name from source column + transform."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", transform)
    return f"{col}_{safe}"


class FuncCommand(BaseCommand):
    name = "func"
    help = "Apply column transforms (cumsum, group mean/sum, qcut, interpolate, …)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = __doc__

        self.add_io_arguments(parser)

        g = parser.add_argument_group("transform options")
        g.add_argument(
            "-c",
            "--col",
            required=True,
            metavar="COL",
            help="Source column to transform.",
        )
        g.add_argument(
            "-t",
            "--transform",
            required=True,
            metavar="TRANSFORM",
            help=(
                "Transform to apply: sum, mean, min, max, count, median, std, "
                "cumsum, rank, pct_rank, qcut:N, interpolate, ffill or bfill."
            ),
        )
        g.add_argument(
            "-d",
            "--destcol",
            default=None,
            metavar="NAME",
            help="Name of the output column (default: <col>_<transform>).",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="Group by these column(s) before applying the transform.",
        )
        g.add_argument(
            "-x",
            "--xcol",
            default=None,
            metavar="COL",
            help="Required for interpolate, ffill, bfill: take each group's rows "
            "in the order of this column, and interpolate against its values.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)

        check_cols(df, [args.col], "-c/--col")
        check_cols(df, args.groupcol, "-g/--groupcol")
        xcol = getattr(args, "xcol", None)
        if xcol is not None:
            check_cols(df, [xcol], "-x/--xcol")

        destcol = args.destcol or _default_destcol(args.col, args.transform)

        df = _apply_transform(
            df, args.col, args.transform, args.groupcol, destcol, xcol
        )

        io.printdf(df, args)
