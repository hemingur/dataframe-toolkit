"""
Tests for dftk.commands.test_cmd._run_test.

Strategy: synthetic datasets with exactly known properties.  Where possible,
expected p-values are computed from scipy directly so tests verify orchestration
(correct grouping, column pairing, output schema) rather than reimplementing
scipy math.  A few tests check directional properties (e.g. identical arrays
yield p≈1, very different arrays yield p<0.05).
"""

import argparse

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from dftk.commands.test_cmd import TESTS, _bootstrap, _run_test

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        cols=["a:b"],
        dest=["pvalue"],
        groups=[],
        test="student_t",
        randomize=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def identical_df():
    """Two identical columns — any two-sample test should yield p≈1."""
    rng = np.random.default_rng(0)
    vals = rng.normal(0, 1, 20)
    return pd.DataFrame({"a": vals, "b": vals.copy()})


@pytest.fixture
def different_df():
    """Two clearly separated columns — parametric tests should yield p<0.05."""
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 30),
            "b": rng.normal(10, 1, 30),
        }
    )


@pytest.fixture
def grouped_df():
    """Two groups, each with a:b pair having known direction."""
    return pd.DataFrame(
        {
            "group": ["X"] * 10 + ["Y"] * 10,
            "a": np.concatenate([np.zeros(10), np.zeros(10)]),
            "b": np.concatenate([np.zeros(10), np.ones(10) * 5]),
        }
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestSchema:
    pytestmark = pytest.mark.filterwarnings(
        "ignore:Precision loss occurred:RuntimeWarning"
    )

    def test_ungrouped_columns(self, different_df):
        result = _run_test(different_df, make_args())
        assert list(result.columns) == ["pvalue"]

    def test_ungrouped_one_row(self, different_df):
        result = _run_test(different_df, make_args())
        assert len(result) == 1

    def test_grouped_columns(self, grouped_df):
        result = _run_test(grouped_df, make_args(groups=["group"]))
        assert list(result.columns) == ["group", "pvalue"]

    def test_grouped_two_rows(self, grouped_df):
        result = _run_test(grouped_df, make_args(groups=["group"]))
        assert len(result) == 2

    def test_grouped_group_values(self, grouped_df):
        result = _run_test(grouped_df, make_args(groups=["group"]))
        assert set(result["group"]) == {"X", "Y"}

    def test_multi_pair_columns(self, different_df):
        df = different_df.rename(columns={"a": "c", "b": "d"}).join(different_df)
        result = _run_test(df, make_args(cols=["a:b", "c:d"], dest=["p1", "p2"]))
        assert list(result.columns) == ["p1", "p2"]
        assert len(result) == 1


# ---------------------------------------------------------------------------
# student_t
# ---------------------------------------------------------------------------


class TestStudentT:
    def test_identical_p_near_1(self, identical_df):
        result = _run_test(identical_df, make_args(test="student_t"))
        assert result["pvalue"].iloc[0] == pytest.approx(1.0)

    def test_different_p_small(self, different_df):
        result = _run_test(different_df, make_args(test="student_t"))
        assert result["pvalue"].iloc[0] < 0.05

    def test_matches_scipy(self, different_df):
        expected = float(
            ss.ttest_ind(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="student_t"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)

    @pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
    def test_grouped_x_identical(self, grouped_df):
        result = _run_test(grouped_df, make_args(test="student_t", groups=["group"]))
        x_row = result[result["group"] == "X"]
        # group X: a=b=0, so t-test cannot be computed (all zeros)
        # scipy returns nan pvalue for zero-variance identical arrays
        assert np.isnan(x_row["pvalue"].iloc[0]) or x_row["pvalue"].iloc[
            0
        ] == pytest.approx(1.0)  # noqa: E501

    @pytest.mark.filterwarnings("ignore:Precision loss occurred:RuntimeWarning")
    def test_grouped_y_different(self, grouped_df):
        result = _run_test(grouped_df, make_args(test="student_t", groups=["group"]))
        y_row = result[result["group"] == "Y"]
        assert y_row["pvalue"].iloc[0] < 0.05


# ---------------------------------------------------------------------------
# paired_student_t
# ---------------------------------------------------------------------------


class TestPairedStudentT:
    def test_identical_p_is_nan(self, identical_df):
        # paired t-test on identical columns: zero differences → NaN pvalue
        result = _run_test(identical_df, make_args(test="paired_student_t"))
        assert np.isnan(result["pvalue"].iloc[0])

    def test_different_p_small(self, different_df):
        result = _run_test(different_df, make_args(test="paired_student_t"))
        assert result["pvalue"].iloc[0] < 0.05

    def test_matches_scipy(self, different_df):
        expected = float(
            ss.ttest_rel(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="paired_student_t"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_different_p_small(self, different_df):
        result = _run_test(different_df, make_args(test="mannwhitneyu"))
        assert result["pvalue"].iloc[0] < 0.05

    def test_matches_scipy(self, different_df):
        expected = float(
            ss.mannwhitneyu(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="mannwhitneyu"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Correlation tests (pearson, spearman, kendall)
# ---------------------------------------------------------------------------


class TestCorrelations:
    @pytest.fixture
    def corr_df(self):
        """Perfectly correlated pair and an uncorrelated pair."""
        x = np.arange(1.0, 21.0)
        return pd.DataFrame(
            {
                "a": x,
                "b": x * 2 + 1,  # perfect positive correlation
                "c": np.random.default_rng(99).permutation(x),  # shuffled
            }
        )

    def test_pearson_perfect_correlation_small_p(self, corr_df):
        result = _run_test(corr_df, make_args(cols=["a:b"], test="pearson"))
        assert result["pvalue"].iloc[0] < 0.001

    def test_pearson_matches_scipy(self, corr_df):
        expected = float(ss.pearsonr(corr_df["a"].values, corr_df["b"].values).pvalue)
        result = _run_test(corr_df, make_args(cols=["a:b"], test="pearson"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)

    def test_spearman_matches_scipy(self, corr_df):
        expected = float(ss.spearmanr(corr_df["a"].values, corr_df["b"].values).pvalue)
        result = _run_test(corr_df, make_args(cols=["a:b"], test="spearman"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)

    def test_kendall_matches_scipy(self, corr_df):
        expected = float(ss.kendalltau(corr_df["a"].values, corr_df["b"].values).pvalue)
        result = _run_test(corr_df, make_args(cols=["a:b"], test="kendall"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov
# ---------------------------------------------------------------------------


class TestKolmogorovSmirnov:
    def test_identical_p_is_1(self, identical_df):
        result = _run_test(identical_df, make_args(test="kolmogorov_smirnov"))
        assert result["pvalue"].iloc[0] == pytest.approx(1.0)

    def test_different_p_small(self, different_df):
        result = _run_test(different_df, make_args(test="kolmogorov_smirnov"))
        assert result["pvalue"].iloc[0] < 0.05

    def test_matches_scipy(self, different_df):
        expected = float(
            ss.ks_2samp(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="kolmogorov_smirnov"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_all_same_sign_p_is_1(self):
        """All differences zero → p = 1."""
        v = np.ones(10)
        assert _bootstrap(v, v) == pytest.approx(1.0)

    def test_clearly_different_p_small(self):
        """a >> b for all samples → nearly all signs agree → p near 0."""
        v1 = np.full(100, 10.0)
        v2 = np.zeros(100)
        assert _bootstrap(v1, v2) < 0.05

    def test_mixed_signs_formula(self):
        """Manually verify the formula: 2 * excess / n."""
        v1 = np.array([1.0, 2.0, -1.0, -2.0])  # diffs: 1, 2, -1, -2
        v2 = np.zeros(4)
        result = _bootstrap(v1, v2)
        # positive=2, negative=2, excess=2, p = 2*2/4 = 1.0
        assert result == pytest.approx(1.0)

    def test_via_run_test(self, different_df):
        result = _run_test(different_df, make_args(test="bootstrap"))
        assert 0.0 <= result["pvalue"].iloc[0] <= 1.0

    def test_randomize_reproducible(self, different_df):
        np.random.seed(7)
        r1 = _run_test(different_df, make_args(test="bootstrap", randomize=True))
        np.random.seed(7)
        r2 = _run_test(different_df, make_args(test="bootstrap", randomize=True))
        assert r1["pvalue"].iloc[0] == r2["pvalue"].iloc[0]


# ---------------------------------------------------------------------------
# Wilcoxon / Kruskal / ANOVA
# ---------------------------------------------------------------------------


class TestOtherTests:
    def test_wilcoxon_different_p_small(self, different_df):
        result = _run_test(different_df, make_args(test="wilcoxon"))
        assert result["pvalue"].iloc[0] < 0.05

    def test_kruskal_matches_scipy(self, different_df):
        expected = float(
            ss.kruskal(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="kruskal"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)

    def test_anova_matches_scipy(self, different_df):
        expected = float(
            ss.f_oneway(different_df["a"].values, different_df["b"].values).pvalue
        )  # noqa: E501
        result = _run_test(different_df, make_args(test="anova"))
        assert result["pvalue"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Multi-pair and edge cases
# ---------------------------------------------------------------------------


class TestMultiPairAndEdgeCases:
    def test_multi_pair_independent(self, different_df):
        """Two pairs produce two independent p-values in the same row."""
        df = different_df.copy()
        df["c"] = df["a"].values
        df["d"] = df["b"].values
        result = _run_test(df, make_args(cols=["a:b", "c:d"], dest=["p1", "p2"]))
        assert result["p1"].iloc[0] == pytest.approx(result["p2"].iloc[0])

    def test_nan_on_bad_input(self):
        """A test that cannot run (e.g. all-NaN column) returns NaN, not crash."""
        df = pd.DataFrame({"a": [float("nan")] * 5, "b": [1.0] * 5})
        result = _run_test(df, make_args(test="student_t"))
        assert np.isnan(result["pvalue"].iloc[0])

    def test_all_tests_registered(self):
        """TESTS list contains all expected test names."""
        expected = {
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
        }
        assert set(TESTS) == expected

    def test_pvalue_in_range(self, different_df):
        """p-value is always in [0, 1] for all implemented tests."""
        for test in TESTS:
            result = _run_test(different_df, make_args(test=test))
            pval = result["pvalue"].iloc[0]
            assert np.isnan(pval) or (0.0 <= pval <= 1.0), f"{test}: pvalue={pval}"
