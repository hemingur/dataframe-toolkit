"""
dftk.commands.corr_cmd — dftk corr subcommand.

Port of dfcorr.py: pairwise column correlations with optional grouping,
analytical confidence intervals (Pearson), and BCa bootstrap inference.

Bootstrap mode (--bootstrap N) produces a single summary row per pair with:
  p_perm      — permutation p-value (fraction of |r_null| >= |r_obs|)
  ci_boot_lo  — BCa 95% CI lower bound (all methods)
  ci_boot_hi  — BCa 95% CI upper bound (all methods)
"""

import argparse
import fnmatch
import itertools
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss

from dftk.commands.base import BaseCommand
from dftk.common.copula import (
    MIN_TAIL_N,
    empirical_tail_dependence,
    pseudo_observations,
)
from dftk.common.io import check_cols, io
from dftk.common.seed import normalize_seed

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
    """Return (p_perm, ci_boot_lo, ci_boot_hi) via permutation test and BCa CI."""
    n = len(a)
    r_obs = method_fn(a, b).statistic

    r_obs_abs = abs(r_obs)
    n_extreme = 0
    for _ in range(n_boot):
        if abs(method_fn(a, rng.permutation(b)).statistic) >= r_obs_abs:
            n_extreme += 1
    p_perm = n_extreme / n_boot

    theta_boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        theta_boot[i] = method_fn(a[idx], b[idx]).statistic

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


def _tail_stats(a: np.ndarray, b: np.ndarray, q_grid: np.ndarray) -> tuple:
    """Return a flat tuple of (lambda_lower_qX, lambda_upper_qX, ...) for each q
    in q_grid, in the same order, plus a stderr warning if any q is too small
    for the sample size to resolve.
    """
    n = len(a)
    u, v = pseudo_observations(a, b)
    lam_L, lam_U = empirical_tail_dependence(u, v, q_grid)
    thin = [q for q in q_grid if n * q < MIN_TAIL_N]
    if thin:
        logger.warning(
            "tail dependence: n=%d gives fewer than %d points in the tail for "
            "q=%s -- these estimates will be noisy. See --tail-q / MIN_TAIL_N.",
            n,
            MIN_TAIL_N,
            ", ".join(f"{q:g}" for q in thin),
        )
    out = []
    for lo, hi in zip(lam_L, lam_U, strict=True):
        out.extend([float(lo), float(hi)])
    return tuple(out)


_GLOB_CHARS = frozenset("*?[")


def _numeric_columns(df: pd.DataFrame, groups: list[str]) -> list[str]:
    """Numeric, non-boolean columns of *df* other than the group columns."""
    return [
        c
        for c in df.columns
        if c not in groups
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def _match(pattern: str, numeric: list[str], option: str) -> list[str]:
    """A column name as is, or the numeric columns a glob pattern matches."""
    if not _GLOB_CHARS & set(pattern):
        return [pattern]
    matched = fnmatch.filter(numeric, pattern)
    if not matched:
        raise ValueError(
            f"{option}: {pattern!r} matches no numeric column (numeric columns: "
            f"{numeric})"
        )
    return matched


def _expand_pairs(cols: list[str], numeric: list[str]) -> list[tuple[str, str]]:
    """-c COL1:COL2 items as pairs; a glob on either side expands.

    A pair from a glob is left out when both sides are the same column or
    when the pair already came up the other way round, so -c '*:*' gives
    every pair once.  Pairs named in full are kept as given.
    """
    bad = [c for c in cols if len(c.split(":", maxsplit=1)) != 2]
    if bad:
        raise ValueError(f"-c pairs must be col1:col2, got: {bad}")
    pairs: list[tuple[str, str]] = []
    for item in cols:
        left, right = item.split(":", maxsplit=1)
        if not _GLOB_CHARS & set(item):
            pairs.append((left, right))
            continue
        for a, b in itertools.product(
            _match(left, numeric, "-c"), _match(right, numeric, "-c")
        ):
            if a != b and (a, b) not in pairs and (b, a) not in pairs:
                pairs.append((a, b))
    return pairs


def _all_pairs(cols: list[str], numeric: list[str]) -> list[tuple[str, str]]:
    """--all [COL ...]: every ordered pair, diagonal included (a square matrix)."""
    names: list[str] = []
    for pattern in cols or ["*"]:
        names += [c for c in _match(pattern, numeric, "--all") if c not in names]
    if len(names) < 2:
        raise ValueError(f"--all needs at least two numeric columns, got {names}")
    return list(itertools.product(names, names))


def _corr(
    df: pd.DataFrame,
    args: argparse.Namespace,
    rng: np.random.Generator,
    pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Correlations for *pairs* (default: args.cols, globs expanded)."""
    if pairs is None:
        pairs = _expand_pairs(args.cols, _numeric_columns(df, args.groups))
    method_fn = _METHOD_FUNC[args.method]

    ci = args.ci and args.method == "pearson"
    tail = getattr(args, "tail", False)
    q_grid = np.asarray(getattr(args, "tail_q", None) or [0.05])

    header = ["col1", "col2", "nobs", "correlation", "pvalue"]
    if ci:
        header += ["cilo", "cihi"]
    if args.bootstrap is not None:
        header += ["p_perm", "ci_boot_lo", "ci_boot_hi"]
    if tail:
        for q in q_grid:
            header += [f"lambda_lower_q{q:g}", f"lambda_upper_q{q:g}"]

    rows = []

    def _add_pairs(sub: pd.DataFrame, prefix: tuple = ()):
        # Every statistic is symmetric in the two columns, so --all's mirrored
        # pair (b, a) reuses the result for (a, b)
        done: dict[frozenset, tuple] = {}
        for col1, col2 in pairs:
            key = frozenset((col1, col2))
            if key not in done:
                if col1 == col2:  # the diagonal of --all; sub[[c, c]] is 2-D
                    a = b = sub[col1].dropna().values
                else:
                    pair_df = sub[[col1, col2]].dropna()
                    a, b = pair_df[col1].values, pair_df[col2].values
                stats = _compute_correlation(a, b, method_fn, ci)
                if col1 == col2:
                    # Bootstrap and tail statistics of a column with itself
                    # mean nothing (ties even push lambda above 1)
                    n_extra = (3 if args.bootstrap is not None else 0) + (
                        2 * len(q_grid) if tail else 0
                    )
                    stats += (np.nan,) * n_extra
                else:
                    if args.bootstrap is not None:
                        stats += _bootstrap_stats(
                            a,
                            b,
                            method_fn,
                            args.bootstrap,
                            rng,
                            args.confidence / 100.0,
                        )
                    if tail:
                        stats += _tail_stats(a, b, q_grid)
                done[key] = stats
            rows.append((*prefix, col1, col2, *done[key]))

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

    dftk corr data.tsv -c x:y
    dftk dataset tips -o | dftk corr -c total_bill:tip tip:size

  A glob on either side expands to the numeric columns it matches (quote
  it). '*:*' gives every pair once; 'sepal*:petal*' every sepal column
  with every petal column:

    dftk dataset iris -o | dftk corr -c '*:*'
    dftk dataset iris -o | dftk corr -c 'sepal*:petal*' --method spearman

CORRELATION MATRIX (--all)
--------------------------
  --all gives every pair both ways round plus the diagonal, in column order:
  the square table a correlogram needs.  Name columns (globs allowed) to
  limit it; -g gives one matrix per group.  Each pair is computed once.  On
  the diagonal r is 1 and the bootstrap and tail columns are empty.

    dftk dataset iris -o | dftk corr --all
    dftk dataset iris -o | dftk corr --all -o \\
      | dftk heatmap -x col1 -y col2 -v correlation --annot
    dftk dataset diamonds -o | dftk corr --all carat depth price -g cut

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

    ci_boot_lo  BCa (bias-corrected and accelerated) CI lower bound.
    ci_boot_hi  BCa CI upper bound.

  The CI level defaults to 95% and is controlled by --confidence.
  BCa corrects for both bias (z0) and skewness (acceleration factor a_hat)
  in the bootstrap distribution.  The acceleration factor is estimated via a
  streaming jackknife: leave-one-out statistics are computed one at a time
  (capped at 500 observations when n > 500) so peak memory is O(n) rather
  than O(n^2).  Falls back to percentile CI if BCa is degenerate.

  Use --randomseed for reproducibility.

TAIL DEPENDENCE (--tail / --tail-q)
------------------------------------
  Adds lambda_lower_qN / lambda_upper_qN columns: nonparametric estimates of
  how likely one variable is to be extreme given the other is, separately
  for the lower and upper tail, at each threshold N given to --tail-q.

  A correlation coefficient describes overall association; it says nothing
  about whether the *extremes* specifically move together more (or less)
  than the bulk of the data. Two pairs can share the same Spearman rho and
  have very different tail behavior -- this is what tells them apart. Two
  markers can look equally correlated genome-wide while only one shares
  its rare, extreme values; two assets can look equally correlated day to
  day while only one actually crashes together.

  Reading it: pass 2-3 values to --tail-q spanning an order of magnitude
  (e.g. --tail-q 0.02 0.05 0.1). If both lambda columns shrink as q shrinks,
  there is no real tail dependence (whatever correlation exists is spread
  evenly, not concentrated in the extremes). If a lambda column stays
  roughly flat across q instead of shrinking, that side has genuine tail
  dependence. A single q in isolation is easy to misread, since even a pair
  with NO tail dependence gives an elevated, nonzero reading at any finite
  q if the pair is strongly correlated overall -- it is the trend across q
  that carries the signal, not any one number.

  This needs enough data in the tail region to be reliable: a warning is
  printed to stderr when nobs * q for the smallest requested q falls below
  20, since anything thinner is dominated by sampling noise. Pair with
  `dftk chi` for a visual, corner-resolved view of the same question.

EXAMPLES
--------
  dftk corr data.tsv -c x:y
  dftk dataset tips -o | dftk corr -c total_bill:tip
  dftk dataset tips -o | dftk corr -c total_bill:tip -g day --method spearman
  dftk dataset tips -o | dftk corr -c total_bill:tip --ci
  dftk dataset tips -o \\
      | dftk corr -c total_bill:tip --method spearman \\
            --bootstrap 2000 --randomseed 42
  dftk dataset diamonds -o | dftk corr -c carat:price --tail --tail-q 0.02 0.05 0.1
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
        which = g.add_mutually_exclusive_group(required=True)
        which.add_argument(
            "-c",
            "--cols",
            nargs="+",
            metavar="COL1:COL2",
            help="Column pair(s) to correlate, e.g. x:y.  A glob on either "
            "side expands to numeric columns: '*:*' is every pair once, "
            "'sepal*:petal*' every sepal column with every petal column.",
        )
        which.add_argument(
            "--all",
            nargs="*",
            default=None,
            metavar="COL",
            help="Correlation matrix: every pair of the named columns (globs "
            "allowed; default: all numeric columns) both ways round, plus the "
            "diagonal, in column order.  Pipe it to dftk heatmap for a "
            "correlogram.",
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
            "--confidence",
            type=float,
            default=95.0,
            metavar="PCT",
            help="CI level in percent for BCa bootstrap bounds (default: 95.0).",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help="Random seed for bootstrap/permutation (integer or string).",
        )
        g.add_argument(
            "--tail",
            action="store_true",
            help="Add nonparametric lower/upper tail-dependence coefficient "
            "columns (see --tail-q).",
        )
        g.add_argument(
            "--tail-q",
            type=float,
            nargs="+",
            default=None,
            metavar="Q",
            help="Tail probability threshold(s) for --tail, e.g. 0.02 0.05 0.1 "
            "(default: 0.05). Smaller q looks deeper into the tail but needs "
            "more data to stay reliable -- see epilog.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        seed = normalize_seed(args.randomseed) if args.randomseed is not None else None
        rng = np.random.default_rng(seed)

        df = io.read(args)
        check_cols(df, args.groups, "-g/--groups")
        numeric = _numeric_columns(df, args.groups)
        if args.all is not None:
            pairs = _all_pairs(args.all, numeric)
            check_cols(df, sorted({c for p in pairs for c in p}), "--all")
        else:
            pairs = _expand_pairs(args.cols, numeric)
            check_cols(df, [c for p in pairs for c in p], "-c/--cols")

        result = _corr(df, args, rng, pairs)
        io.printdf(result, args)
