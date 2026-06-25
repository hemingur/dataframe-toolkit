"""
stattools.commands.corr_cmd — dfstat corr subcommand.

Port of dfcorr.py: pairwise column correlations with optional grouping,
analytical confidence intervals (Pearson), and BCa bootstrap inference.

Bootstrap mode (--bootstrap N) produces a single summary row per pair with:
  p_boot      — permutation p-value (fraction of |r_null| >= |r_obs|)
  ci_boot_lo  — BCa 95% CI lower bound (all methods)
  ci_boot_hi  — BCa 95% CI upper bound (all methods)
"""

import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io
from stattools.common.seed import normalize_seed

logger = logging.getLogger(__name__)

METHODS = ["pearson", "spearman", "kendall"]
_METHOD_FUNC = {
    "pearson": ss.pearsonr,
    "spearman": ss.spearmanr,
    "kendall": ss.kendalltau,
}


def _compute_correlation(a: np.ndarray, b: np.ndarray, method_fn, ci: bool) -> tuple:
    result = method_fn(a, b)
    row = (len(a), result.statistic, result.pvalue)
    if ci:
        interval = result.confidence_interval()
        row = (*row, interval.low, interval.high)
    return row


def _bootstrap_stats(
    a: np.ndarray,
    b: np.ndarray,
    method_fn,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (p_boot, ci_boot_lo, ci_boot_hi).

    p_boot uses a permutation test (shuffle b, build null distribution).
    CIs use BCa bootstrap on paired resamples (falls back to percentile
    if BCa cannot be computed, e.g. zero-variance statistic).
    """
    r_obs = abs(method_fn(a, b).statistic)

    # Permutation p-value: shuffle b to break pairing
    n_extreme = sum(
        1 for _ in range(n) if abs(method_fn(a, rng.permutation(b)).statistic) >= r_obs
    )
    p_boot = n_extreme / n

    # BCa CI via scipy.stats.bootstrap on paired data
    def _stat(x, y):
        return method_fn(x, y).statistic

    for bca_method in ("BCa", "percentile"):
        try:
            boot = ss.bootstrap(
                (a, b),
                statistic=_stat,
                n_resamples=n,
                method=bca_method,
                paired=True,
                vectorized=False,
                random_state=rng,
            )
            ci_lo = boot.confidence_interval.low
            ci_hi = boot.confidence_interval.high
            if np.isnan(ci_lo) or np.isnan(ci_hi):
                raise ValueError("degenerate CI")
            if bca_method == "percentile":
                logger.warning("BCa CI failed; fell back to percentile method.")
            break
        except Exception:  # noqa: BLE001
            continue
    else:
        ci_lo = ci_hi = float("nan")

    return p_boot, ci_lo, ci_hi


def _corr(
    df: pd.DataFrame, args: argparse.Namespace, rng: np.random.Generator
) -> pd.DataFrame:
    method_fn = _METHOD_FUNC[args.method]
    pairs = [c.split(":", maxsplit=1) for c in args.cols]

    ci = args.ci and args.method == "pearson"

    header = ["col1", "col2", "nobs", "correlation", "pvalue"]
    if ci:
        header += ["cilo", "cihi"]
    if args.bootstrap is not None:
        header += ["p_boot", "ci_boot_lo", "ci_boot_hi"]

    rows = []

    def _add_pairs(sub: pd.DataFrame, prefix: tuple = ()):
        for col1, col2 in pairs:
            pair_df = sub[[col1, col2]].dropna()
            a, b = pair_df[col1].values, pair_df[col2].values
            cr = _compute_correlation(a, b, method_fn, ci)
            if args.bootstrap is None:
                rows.append((*prefix, col1, col2, *cr))
            else:
                boot = _bootstrap_stats(a, b, method_fn, args.bootstrap, rng)
                rows.append((*prefix, col1, col2, *cr, *boot))

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
  Adds cilo and cihi columns using the analytical formula.
  Only available for Pearson; silently ignored for other methods.
  For Spearman/Kendall use --bootstrap to get BCa CIs.

BOOTSTRAP (--bootstrap N)
--------------------------
  Adds three summary columns to the (single) output row per pair:
    p_boot      Non-parametric p-value via permutation test.
    ci_boot_lo  BCa 95% CI lower bound (all methods).
    ci_boot_hi  BCa 95% CI upper bound (all methods).
  Use --randomseed for reproducibility.

EXAMPLES
--------
  dfstat corr data.tsv -c x:y
  dfstat corr data.tsv -c x:y -g group --method spearman
  dfstat corr data.tsv -c x:y --ci
  dfstat corr data.tsv -c x:y --method spearman --bootstrap 2000 --randomseed 42
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
            help="Include analytical 95%% CI columns cilo/cihi (Pearson only).",
        )
        g.add_argument(
            "--bootstrap",
            type=int,
            default=None,
            metavar="N",
            help="Bootstrap N resamples; adds p_boot and BCa CI columns.",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help="Random seed for bootstrap (integer or string).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        seed = normalize_seed(args.randomseed) if args.randomseed is not None else None
        rng = np.random.default_rng(seed)

        all_cols = [c for pair in args.cols for c in pair.split(":", maxsplit=1)]
        df = io.read(args)
        check_cols(df, all_cols, "-c/--cols")
        check_cols(df, args.groups, "-g/--groups")

        result = _corr(df, args, rng)
        io.printdf(result, args)
