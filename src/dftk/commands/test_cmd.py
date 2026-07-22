"""
dftk.commands.test_cmd — the ``dftk test`` subcommand.

Computes p-values for statistical tests between pairs of columns,
optionally within groups.  Each column pair ``col1:col2`` produces one
output column named by the corresponding ``--dest`` entry.

Tests available
---------------
  bootstrap           Sign-based p-value for paired bootstrap differences
  student_t           Welch's independent-samples t-test
  paired_student_t    Paired t-test
  anova               One-way analysis of variance (F-test)
  pearson             Pearson product-moment correlation
  kendall             Kendall rank correlation τ
  spearman            Spearman rank-order correlation
  mannwhitneyu        Mann-Whitney U rank test
  wilcoxon            Wilcoxon signed-rank test
  kruskal             Kruskal-Wallis H-test
  kolmogorov_smirnov  Two-sample Kolmogorov-Smirnov test

The bootstrap test counts sign agreements between paired differences.
``--randomize`` shuffles the second column before pairing, which provides
a permutation-based null distribution.  Use ``--randomseed`` for
reproducibility.

Example
-------
    dftk test data.tsv -c col1:col2 -d pvalue --test student_t
    dftk test data.tsv -c a:b c:d -d p_ab p_cd -g group --test mannwhitneyu
    dftk test boot.tsv -c s1:s2 -d p --test bootstrap --randomseed 42
"""

import argparse
import logging

import numpy as np
import pandas as pd
import scipy.stats as ss

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io
from dftk.common.seed import normalize_seed

logger = logging.getLogger("dftk")

# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

TESTS = [
    "bootstrap",
    "student_t",
    "paired_student_t",
    "anova",
    "pearson",
    "kendall",
    "spearman",
    "mannwhitneyu",
    "wilcoxon",
    "kruskal",
    "kolmogorov_smirnov",
]


def _bootstrap(v1: np.ndarray, v2: np.ndarray, randomize: bool = False) -> float:
    """Sign-based paired bootstrap p-value."""
    if randomize:
        v2 = np.random.permutation(v2)
    diffs = v1 - v2
    signs = np.sign(diffs)
    nonzero = np.nonzero(signs)[0]
    if nonzero.size == 0:
        return 1.0
    positive = int(np.sum(signs > 0))
    negative = int(np.sum(signs < 0))
    excess = min(positive, negative)
    return 2 * excess / len(diffs)


def _pvalue(v1: np.ndarray, v2: np.ndarray, test: str, randomize: bool) -> float:
    """Return the p-value for *test* applied to arrays *v1* and *v2*."""
    if test == "bootstrap":
        return _bootstrap(v1, v2, randomize)
    fn = {
        "student_t": ss.ttest_ind,
        "paired_student_t": ss.ttest_rel,
        "anova": ss.f_oneway,
        "pearson": ss.pearsonr,
        "kendall": ss.kendalltau,
        "spearman": ss.spearmanr,
        "mannwhitneyu": ss.mannwhitneyu,
        "wilcoxon": ss.wilcoxon,
        "kruskal": ss.kruskal,
        "kolmogorov_smirnov": ss.ks_2samp,
    }[test]
    return float(fn(v1, v2).pvalue)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _run_test(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return a DataFrame of p-values for one or more column pairs.

    *args* must have:
      - ``cols``      list of ``"col1:col2"`` strings
      - ``dest``      list of output column names, same length as cols
      - ``groups``    list of group-by column names, or empty list / None
      - ``test``      test name (one of TESTS)
      - ``randomize`` bool — shuffle second column before bootstrap pairing
    """
    col_pairs = [x.split(":", 1) for x in args.cols]
    dest = args.dest
    groups = args.groups or []
    test = args.test
    randomize = getattr(args, "randomize", False)

    def _row_values(sub: pd.DataFrame) -> list:
        row = []
        for col1, col2 in col_pairs:
            try:
                row.append(_pvalue(sub[col1].values, sub[col2].values, test, randomize))
            except Exception as exc:
                logger.warning("test failed for %s:%s — %s", col1, col2, exc)
                row.append(float("nan"))
        return row

    if not groups:
        return pd.DataFrame([_row_values(df)], columns=dest)

    results = []
    for group_key, group_df in df.groupby(groups):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        results.append(list(group_key) + _row_values(group_df))

    return pd.DataFrame(results, columns=groups + dest)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class TestCommand(BaseCommand):
    """Run statistical tests between column pairs, optionally within groups."""

    name = "test"
    help = (
        "Compute p-values between column pairs "
        "(t-test, Mann-Whitney, bootstrap, …); supports grouping"
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.add_io_arguments(parser)

        g = parser.add_argument_group("test")
        g.add_argument(
            "-c",
            "--cols",
            help="Column pair(s) to test, in the form col1:col2",
            nargs="+",
            required=True,
            metavar="COL1:COL2",
        )
        g.add_argument(
            "-d",
            "--dest",
            help="Output column name(s) for p-values (one per --cols pair)",
            nargs="+",
            required=True,
            metavar="NAME",
        )
        g.add_argument(
            "-g",
            "--groups",
            help="Group-by column(s); test is run within each group",
            nargs="+",
            default=[],
            metavar="COL",
        )
        g.add_argument(
            "--test",
            help="Statistical test to use (default: bootstrap)",
            choices=TESTS,
            default="bootstrap",
        )
        g.add_argument(
            "--tags",
            help="Extra tag columns appended to output, in tag:value format",
            nargs="+",
            default=None,
            metavar="TAG:VALUE",
        )

        bs = parser.add_argument_group("bootstrap options")
        bs.add_argument(
            "--randomize",
            help="Shuffle second column before pairing (bootstrap only)",
            action="store_true",
        )
        bs.add_argument(
            "--randomseed",
            help="Random seed — integer or string (MD5-hashed to integer)",
            default=None,
            metavar="SEED",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if len(args.cols) != len(args.dest):
            raise ValueError(
                f"--cols has {len(args.cols)} pair(s) but --dest has "
                f"{len(args.dest)} name(s); counts must match"
            )

        args.randomseed = normalize_seed(args.randomseed)
        np.random.seed(args.randomseed)

        df = io.read(args)

        all_cols = [c for pair in args.cols for c in pair.split(":", 1)]
        check_cols(df, all_cols, "-c/--cols")
        check_cols(df, args.groups, "-g/--groups")

        result = _run_test(df, args)

        if args.tags:
            for item in args.tags:
                tag, val = item.split(":", 1)
                result[tag] = val

        io.printdf(result, args)
