"""
stattools.commands.corr_cmd — dfstat corr subcommand.

Port of dfcorr.py: pairwise column correlations with optional grouping,
analytical confidence intervals (Pearson), and BCa bootstrap inference.

Bootstrap mode (--bootstrap N) produces a single summary row per pair with:
  p_perm      — permutation p-value (fraction of |r_null| >= |r_obs|)
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


# BCa jackknife acceleration is estimated from leave-one-out resamples.
# scipy.stats.bootstrap materialises an (n × n-1) array for this step,
# which OOMs at large n.  We stream one leave-one-out at a time and cap
# at _MAX_JACK observations so peak memory is O(min(n, _MAX_JACK)).
_MAX_JACK = 500


def _bootstrap_stats(
    a: np.ndarray,
    b: np.ndarray,
    method_fn,
    n_boot: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Return (p_perm, ci_boot_lo, ci_boot_hi).

    p_perm:       permutation p-value (fraction of |r_null| >= |r_obs|).
    ci_boot_lo/hi: BCa 95% CI with streaming jackknife (O(n) peak memory).
                  Falls back to percentile CI when BCa is degenerate.
    """
    n = len(a)
    r_obs = method_fn(a, b).statistic

    # --- permutation test ---------------------------------------------------
    r_obs_abs = abs(r_obs)
    n_extreme = 0
    for _ in range(n_boot):
        if abs(method_fn(a, rng.permutation(b)).statistic) >= r_obs_abs:
            n_extreme += 1
    p_perm = n_extreme / n_boot

    # --- bootstrap distribution (resample with replacement) -----------------
    theta_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        theta_boot[i] = method_fn(a[idx], b[idx]).statistic

    # --- BCa acceleration via streaming jackknife ---------------------------
    if n <= _MAX_JACK:
        jack_indices = np.arange(n)
    else:
        jack_indices = rng.choice(n, size=_MAX_JACK, replace=False)

    theta_jack = np.empty(len(jack_indices))
    mask = np.ones(n, dtype=bool)
    for j, i in enumerate(jack_indices):
        mask[i] = False
        theta_jack[j] = method_fn(a[mask], b[mask]).statistic
        mask[i] = True

    jmean = theta_jack.mean()
    diff = jmean - theta_jack
    denom = np.sum(diff**2)
    a_hat = np.sum(diff**3) / (6.0 * denom**1.5) if denom > 1e-20 else 0.0

    # --- BCa bias correction ------------------------------------------------
    prop = np.clip((theta_boot < r_obs).mean(), 1e-10, 1 - 1e-10)
    z0 = ss.norm.ppf(prop)

    alpha = (1.0 - confidence) / 2.0
    z_lo, z_hi = ss.norm.ppf(alpha), ss.norm.ppf(1.0 - alpha)

    def _adj(z: float) -> float:
        denom = 1.0 - a_hat * (z0 + z)
        if abs(denom) < 1e-10:
            return float("nan")
        return ss.norm.cdf(z0 + (z0 + z) / denom)

    a1, a2 = _adj(z_lo), _adj(z_hi)

    if np.isnan(a1) or np.isnan(a2):
        ci_lo = float(np.percentile(theta_boot, 100.0 * alpha))
        ci_hi = float(np.percentile(theta_boot, 100.0 * (1.0 - alpha)))
    else:
        ci_lo = float(np.percentile(theta_boot, 100.0 * a1))
        ci_hi = float(np.percentile(theta_boot, 100.0 * a2))

    return p_perm, ci_lo, ci_hi


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
        header += ["p_perm", "ci_boot_lo", "ci_boot_hi"]

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
  Adds cilo and cihi columns using the analytical Fisher z-transform formula.
  Only available for Pearson; silently ignored for Spearman/Kendall.
  Use --bootstrap to get non-parametric CIs for any method.

BOOTSTRAP (--bootstrap N)
--------------------------
  Runs N bootstrap resamples and N permutation draws, adding three columns:

    p_perm      Permutation p-value: fraction of |r_null| >= |r_obs| when b
                is repeatedly shuffled to break the pairing with a.  This is
                a non-parametric test of the null hypothesis r = 0.

    ci_boot_lo  BCa (bias-corrected and accelerated) 95% CI lower bound.
    ci_boot_hi  BCa 95% CI upper bound.

  BCa corrects for both bias (z0) and skewness (acceleration factor a_hat)
  in the bootstrap distribution.  The acceleration factor is estimated via a
  streaming jackknife: leave-one-out statistics are computed one at a time
  (capped at 500 observations when n > 500) so peak memory is O(n) rather
  than O(n^2).  Falls back to percentile CI if BCa is degenerate.

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
            help="Bootstrap N resamples; adds p_perm and BCa CI columns.",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help="Random seed for bootstrap/permutation (integer or string).",
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
