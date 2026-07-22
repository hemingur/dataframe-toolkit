"""
dftk.commands.stat_cmd — the ``dftk stat`` subcommand.

Computes descriptive statistics for one or more numeric columns, optionally
within groups.  Output always includes a ``name`` column identifying which
input column each row describes, giving a consistent schema regardless of
how many columns are requested.

Output columns
--------------
  [groupcol(s),] name, count, sum, mean, std, min, max, median,
  cilo, cihi, sem, skew, kurt

Bootstrap mode
--------------
With ``--bootstrap N`` the full stat computation is repeated N times on
resampled data.  Each iteration appends a ``samplenum`` column and output
blocks are streamed without repeating the header, making it easy to pipe
into ``dftk pivot`` to aggregate across samples.

Example
-------
    dftk stat data.tsv -c height weight -g sex

    # Empirical bootstrap CI: resample, then collapse with percentile aggfuncs
    dftk stat data.tsv -c value -g group --bootstrap 1000 --randomseed 42 -o \\
        | dftk pivot ... -i group -v mean -f mean cilo cihi -o \\
        | dftk print ...
"""

import argparse
import logging

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io
from dftk.common.seed import normalize_seed

logger = logging.getLogger("dftk")

_CI_METHODS = ["linear", "lower", "higher", "midpoint", "nearest"]

# Canonical order of statistic columns in the output.
_STAT_NAMES = [
    "count",
    "sum",
    "mean",
    "std",
    "min",
    "max",
    "median",
    "cilo",
    "cihi",
    "sem",
    "skew",
    "kurt",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _samplefraction(x: str) -> float:
    """argparse type validator for --samplefrac."""
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a valid float") from None
    if not (0.0 <= x <= 1.0):
        raise argparse.ArgumentTypeError(f"{x} is not in range [0.0, 1.0]")
    return x


def _ci(name: str, level: float, method: str):
    """Return a named percentile aggregation function for use with pandas agg.

    The function name is set explicitly so pandas uses it as the output
    column name rather than the generic ``<lambda>`` or ``cifunction``.
    """

    def cifunction(a):
        return np.percentile(a, q=level, method=method)

    cifunction.__name__ = name
    return cifunction


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _compute_stats(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return a dataframe of descriptive statistics.

    Always produces a ``name`` column identifying the source column, giving
    a consistent schema regardless of how many columns are in ``args.cols``.
    """
    cols: list[str] = args.cols
    groupcol: list[str] | None = args.groupcol

    conf_lo = (100.0 - args.confidencelevel) / 2
    conf_hi = 100.0 - conf_lo
    cilo_fn = _ci("cilo", conf_lo, args.confidencemethod)
    cihi_fn = _ci("cihi", conf_hi, args.confidencemethod)

    agg_funcs = [
        "count",
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "median",
        cilo_fn,
        cihi_fn,
        "sem",
        "skew",
        "kurt",
    ]

    if groupcol is None:
        # Ungrouped — agg returns shape (n_stats, n_cols); transpose to
        # (n_cols, n_stats) then reset the column index to get a 'name' row.
        stats = (
            df[cols].agg(agg_funcs).T.reset_index().rename(columns={"index": "name"})
        )
        return stats[["name"] + _STAT_NAMES]

    else:
        # Grouped — agg returns MultiIndex columns (col, stat).
        # Reshape to long format so every (group, col) combination is a row.
        name_col = "name"
        if name_col in groupcol:
            name_col = "name_mangled"
            logger.warning(
                'A group column is named "name"; '
                'stat name column renamed to "name_mangled"'
            )

        stats = (
            df.groupby(groupcol)[cols]
            .agg(agg_funcs)
            .rename(columns={"amin": "min", "amax": "max"})  # numpy compat
        )

        # swaplevel: (col, stat) → (stat, col)
        # stack:     fold the col level into the row index
        # reset_index + rename: flatten to a plain DataFrame
        level_col = f"level_{len(groupcol)}"
        stats = (
            stats.swaplevel(0, 1, axis=1)
            .stack(future_stack=True)
            .reset_index()
            .rename(columns={level_col: name_col})
            .sort_values(by=[name_col] + groupcol)
        )

        out_cols = groupcol + [name_col] + _STAT_NAMES
        return stats[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class StatCommand(BaseCommand):
    """Compute descriptive statistics for one or more columns."""

    name = "stat"
    help = "Compute descriptive statistics (mean, std, CI, …) for one or more columns"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.add_io_arguments(parser)

        g = parser.add_argument_group("statistics")
        g.add_argument(
            "-c",
            "--cols",
            help="Column(s) to compute statistics on",
            nargs="+",
            required=True,
            metavar="COL",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            help="Group-by column(s)",
            nargs="+",
            default=None,
            metavar="COL",
        )
        g.add_argument(
            "--confidencelevel",
            help="Confidence level in percent (default: 95.0)",
            type=float,
            default=95.0,
            metavar="PCT",
        )
        g.add_argument(
            "--confidencemethod",
            help="Percentile interpolation method (default: linear)",
            choices=_CI_METHODS,
            default="linear",
        )

        bs = parser.add_argument_group("bootstrap")
        bs.add_argument(
            "--bootstrap",
            help="Number of bootstrap samples to draw",
            type=int,
            default=None,
            metavar="N",
        )
        bs.add_argument(
            "--randomseed",
            help=(
                "Random seed — integer, or a string which is MD5-hashed "
                "to a reproducible integer"
            ),
            default=None,
            metavar="SEED",
        )
        size_grp = bs.add_mutually_exclusive_group()
        size_grp.add_argument(
            "--samplesize",
            help="Rows per bootstrap sample (default: same size as input)",
            type=int,
            default=None,
            metavar="N",
        )
        size_grp.add_argument(
            "--samplefrac",
            help="Fraction of rows per bootstrap sample, 0.0–1.0 (default: 1.0)",
            type=_samplefraction,
            default=1.0,
            metavar="FRAC",
        )
        bs.add_argument(
            "--noreplace",
            help="Sample without replacement (default: with replacement)",
            dest="replace",
            action="store_false",
        )
        parser.set_defaults(replace=True)

    def execute(self, args: argparse.Namespace) -> None:
        args.randomseed = normalize_seed(args.randomseed)
        np.random.seed(args.randomseed)

        df = io.read(args)
        if df.empty:
            raise ValueError("Input data is empty")

        check_cols(df, args.cols, "-c/--cols")
        check_cols(df, args.groupcol, "-g/--groupcol")

        if args.bootstrap is None:
            result = _compute_stats(df, args)
            io.printdf(result, args)

        else:
            logger.info(
                f"Bootstrapping: {args.bootstrap} samples, "
                f"seed={args.randomseed}, replace={args.replace}"
            )
            sample_kwargs: dict = {"replace": args.replace}
            if args.samplesize is not None:
                sample_kwargs["n"] = args.samplesize
            else:
                sample_kwargs["frac"] = args.samplefrac

            frames = []
            for i in range(1, args.bootstrap + 1):
                if args.groupcol is None:
                    sample = df.sample(**sample_kwargs)
                else:
                    # [df.columns.tolist()] re-selection works around a pandas 3.0
                    # groupby(...).apply() change that otherwise drops the
                    # grouping column from the result.
                    sample = df.groupby(args.groupcol, group_keys=False)[
                        df.columns.tolist()
                    ].apply(lambda x: x.sample(**sample_kwargs))

                result = _compute_stats(sample, args)
                result["samplenum"] = i
                frames.append(result)

            io.printdf(pd.concat(frames, ignore_index=True), args)
