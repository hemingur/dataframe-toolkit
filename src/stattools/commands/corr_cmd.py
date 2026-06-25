"""
stattools.commands.corr_cmd — dfstat corr subcommand.

Port of dfcorr.py: pairwise column correlations with optional grouping,
confidence intervals (Pearson only), and bootstrap resampling.
"""

import argparse

import numpy as np
import pandas as pd
import scipy.stats as ss

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io
from stattools.common.seed import normalize_seed

METHODS = ["pearson", "spearman", "kendall"]
_METHOD_FUNC = {
    "pearson": ss.pearsonr,
    "spearman": ss.spearmanr,
    "kendall": ss.kendalltau,
}


def _compute_correlation(
    df: pd.DataFrame, col1: str, col2: str, method_fn, ci: bool
) -> tuple:
    result = method_fn(df[col1].values, df[col2].values)
    row = (col1, col2, len(df), result.statistic, result.pvalue)
    if ci:
        interval = result.confidence_interval()
        row = (*row, interval.low, interval.high)
    return row


def _corr(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    method_fn = _METHOD_FUNC[args.method]
    pairs = [c.split(":", maxsplit=1) for c in args.cols]

    ci = args.ci and args.method == "pearson"

    header = ["col1", "col2", "nobs", "correlation", "pvalue"]
    if ci:
        header += ["cilo", "cihi"]
    if args.bootstrap is not None:
        header += ["samplenum"]

    rows = []

    def _add_pairs(sub: pd.DataFrame, prefix: tuple = ()):
        for col1, col2 in pairs:
            pair_df = sub[[col1, col2]].dropna()
            if args.bootstrap is None:
                rows.append(
                    (*prefix, *_compute_correlation(pair_df, col1, col2, method_fn, ci))
                )
            else:
                for i in range(args.bootstrap):
                    sample = pair_df.sample(frac=1.0, replace=True)
                    cr = _compute_correlation(sample, col1, col2, method_fn, ci)
                    rows.append((*prefix, *cr, i))

    if args.groups:
        header = [*args.groups, *header]
        for group, gdf in df.groupby(args.groups):
            prefix = group if isinstance(group, tuple) else (group,)
            _add_pairs(gdf, prefix)
    else:
        _add_pairs(df)

    return pd.DataFrame(rows, columns=header)


_EPILOG = """\
COLUMN PAIRS (-c)
-----------------
  Pairs are specified as col1:col2.  Multiple pairs may be given:

    dfstat corr data.tsv -c x:y
    dfstat corr data.tsv -c x:y a:b

METHODS (--method)
------------------
  pearson    Pearson r (default)
  spearman   Spearman rank correlation
  kendall    Kendall tau

CONFIDENCE INTERVAL (--ci)
--------------------------
  Adds cilo and cihi columns.  Only available for Pearson.
  For other methods the flag is silently ignored.

BOOTSTRAP (--bootstrap N)
--------------------------
  Resamples with replacement N times.  Each resample produces one row
  with a samplenum column.  Use --randomseed for reproducibility.

EXAMPLES
--------
  dfstat corr data.tsv -c x:y
  dfstat corr data.tsv -c x:y -g group --method spearman
  dfstat corr data.tsv -c x:y --ci
  dfstat corr data.tsv -c x:y --bootstrap 1000 --randomseed 42
"""


class CorrCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "corr"

    @property
    def help(self) -> str:
        return "Pairwise column correlations (Pearson, Spearman, Kendall) with optional bootstrap."  # noqa: E501

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("corr options")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            required=True,
            metavar="COL1:COL2",
            help="Column pair(s) to correlate, e.g. x:y.",
        )
        g.add_argument(
            "-g",
            "--groups",
            nargs="+",
            default=[],
            metavar="COL",
            help="Group column(s); correlation is computed within each group.",
        )
        g.add_argument(
            "--method",
            choices=METHODS,
            default="pearson",
            help="Correlation method (default: pearson).",
        )
        g.add_argument(
            "--ci",
            action="store_true",
            help="Include 95%% confidence interval columns cilo/cihi (Pearson only).",
        )
        g.add_argument(
            "--bootstrap",
            type=int,
            default=None,
            metavar="N",
            help="Bootstrap N resamples; adds samplenum column.",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help="Random seed for bootstrap (integer or string).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.randomseed is not None or args.bootstrap is not None:
            np.random.seed(normalize_seed(args.randomseed))

        all_cols = [c for pair in args.cols for c in pair.split(":", maxsplit=1)]
        df = io.read(args)
        check_cols(df, all_cols, "-c/--cols")
        check_cols(df, args.groups, "-g/--groups")

        result = _corr(df, args)
        io.printdf(result, args)
