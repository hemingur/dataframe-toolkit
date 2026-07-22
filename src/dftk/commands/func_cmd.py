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

EXAMPLES
--------
Cumulative sum of "value":

  dftk func data.tsv -c value -t cumsum

Group mean broadcast (adds "value_mean" unless -d is given):

  dftk func data.tsv -c value -g group -t mean

Quantile bins (quartiles):

  dftk func data.tsv -c score -t qcut:4 -d score_quartile

Multiple transforms in one call:

  dftk func data.tsv -c expr -t cumsum -d expr_cumsum \\
      | dftk func - -c expr -t mean -g condition -d expr_grpmean
"""

import argparse
import re

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

# Transforms that use groupby().transform()
_GROUPBY_TRANSFORMS = {"sum", "mean", "min", "max", "count", "median", "std"}

# Transforms that use cumsum / rank within an optional group
_CUMULATIVE_TRANSFORMS = {"cumsum", "rank", "pct_rank"}


def _apply_transform(
    df: pd.DataFrame,
    col: str,
    transform: str,
    groupcols: list[str] | None,
    destcol: str,
) -> pd.DataFrame:
    """Apply *transform* on *col* within optional *groupcols*, storing result in *destcol*."""  # noqa: E501

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

    raise ValueError(
        f"Unknown transform {transform!r}.  "
        "Valid: sum, mean, min, max, count, median, std, cumsum, rank, pct_rank, qcut:N"
    )


def _default_destcol(col: str, transform: str) -> str:
    """Generate a destination column name from source column + transform."""
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", transform)
    return f"{col}_{safe}"


class FuncCommand(BaseCommand):
    name = "func"
    help = "Apply column transforms (cumsum, group mean/sum, qcut, …)."

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
                "cumsum, rank, pct_rank, or qcut:N."
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

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)

        check_cols(df, [args.col], "-c/--col")
        check_cols(df, args.groupcol, "-g/--groupcol")

        destcol = args.destcol or _default_destcol(args.col, args.transform)

        df = _apply_transform(df, args.col, args.transform, args.groupcol, destcol)

        io.printdf(df, args)
