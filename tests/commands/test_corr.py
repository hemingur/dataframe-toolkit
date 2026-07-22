"""Tests for dftk.commands.corr_cmd._corr and _bootstrap_stats."""

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from dftk.commands.corr_cmd import METHODS, _bootstrap_stats, _corr
from tests.conftest import make_args


def _rng(seed=0):
    return np.random.default_rng(seed)


def _args(**kwargs):
    defaults = dict(
        cols=["a:b"],
        groups=[],
        method="pearson",
        ci=False,
        bootstrap=None,
        confidence=95.0,
        randomseed=None,
    )
    defaults.update(kwargs)
    return make_args(**defaults)


@pytest.fixture
def perfect_df():
    x = np.arange(1.0, 21.0)
    return pd.DataFrame({"a": x, "b": x * 2 + 1})


@pytest.fixture
def uncorr_df():
    rng = np.random.default_rng(7)
    n = 30
    return pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})


@pytest.fixture
def grouped_df():
    rng = np.random.default_rng(42)
    n = 20
    x = np.arange(1.0, n + 1)
    return pd.DataFrame(
        {
            "g": ["A"] * n + ["B"] * n,
            "a": np.concatenate([x, rng.permutation(x)]),
            "b": np.concatenate([x * 2, rng.permutation(x)]),
        }
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_columns_no_ci_no_bootstrap(self, perfect_df):
        result = _corr(perfect_df, _args(), _rng())
        assert list(result.columns) == ["col1", "col2", "nobs", "correlation", "pvalue"]

    def test_one_row_one_pair(self, perfect_df):
        result = _corr(perfect_df, _args(), _rng())
        assert len(result) == 1

    def test_col1_col2_values(self, perfect_df):
        result = _corr(perfect_df, _args(), _rng())
        assert result["col1"].iloc[0] == "a"
        assert result["col2"].iloc[0] == "b"

    def test_nobs(self, perfect_df):
        result = _corr(perfect_df, _args(), _rng())
        assert result["nobs"].iloc[0] == len(perfect_df)

    def test_ci_columns_added(self, perfect_df):
        result = _corr(perfect_df, _args(ci=True), _rng())
        assert "cilo" in result.columns
        assert "cihi" in result.columns

    def test_ci_ignored_for_spearman(self, perfect_df):
        result = _corr(perfect_df, _args(method="spearman", ci=True), _rng())
        assert "cilo" not in result.columns

    def test_bootstrap_columns_added(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=50), _rng())
        assert "p_perm" in result.columns
        assert "ci_boot_lo" in result.columns
        assert "ci_boot_hi" in result.columns

    def test_bootstrap_one_row_per_pair(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=50), _rng())
        assert len(result) == 1

    def test_multi_pair_rows(self, perfect_df):
        df = perfect_df.rename(columns={"a": "c", "b": "d"}).join(perfect_df)
        result = _corr(df, _args(cols=["a:b", "c:d"]), _rng())
        assert len(result) == 2

    def test_grouped_columns(self, grouped_df):
        result = _corr(grouped_df, _args(groups=["g"]), _rng())
        assert list(result.columns[:1]) == ["g"]
        assert "correlation" in result.columns

    def test_grouped_two_rows(self, grouped_df):
        result = _corr(grouped_df, _args(groups=["g"]), _rng())
        assert len(result) == 2

    def test_grouped_group_values(self, grouped_df):
        result = _corr(grouped_df, _args(groups=["g"]), _rng())
        assert set(result["g"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# Pearson
# ---------------------------------------------------------------------------


class TestPearson:
    def test_perfect_correlation_is_1(self, perfect_df):
        result = _corr(perfect_df, _args(method="pearson"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(1.0)

    def test_matches_scipy(self, perfect_df):
        expected = ss.pearsonr(perfect_df["a"].values, perfect_df["b"].values)
        result = _corr(perfect_df, _args(method="pearson"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(expected.statistic)
        assert result["pvalue"].iloc[0] == pytest.approx(expected.pvalue)

    def test_ci_bounds(self, perfect_df):
        result = _corr(perfect_df, _args(ci=True), _rng())
        cilo = result["cilo"].iloc[0]
        cihi = result["cihi"].iloc[0]
        corr = result["correlation"].iloc[0]
        assert cilo <= corr <= cihi

    def test_ci_matches_scipy(self, perfect_df):
        expected = ss.pearsonr(perfect_df["a"].values, perfect_df["b"].values)
        interval = expected.confidence_interval()
        result = _corr(perfect_df, _args(ci=True), _rng())
        assert result["cilo"].iloc[0] == pytest.approx(interval.low)
        assert result["cihi"].iloc[0] == pytest.approx(interval.high)


# ---------------------------------------------------------------------------
# Spearman / Kendall
# ---------------------------------------------------------------------------


class TestSpearmanKendall:
    def test_spearman_perfect(self, perfect_df):
        result = _corr(perfect_df, _args(method="spearman"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(1.0)

    def test_spearman_matches_scipy(self, perfect_df):
        expected = ss.spearmanr(perfect_df["a"].values, perfect_df["b"].values)
        result = _corr(perfect_df, _args(method="spearman"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(expected.statistic)

    def test_kendall_perfect(self, perfect_df):
        result = _corr(perfect_df, _args(method="kendall"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(1.0)

    def test_kendall_matches_scipy(self, perfect_df):
        expected = ss.kendalltau(perfect_df["a"].values, perfect_df["b"].values)
        result = _corr(perfect_df, _args(method="kendall"), _rng())
        assert result["correlation"].iloc[0] == pytest.approx(expected.statistic)

    def test_all_methods_registered(self):
        assert set(METHODS) == {"pearson", "spearman", "kendall"}


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_a_high_correlation(self, grouped_df):
        result = _corr(grouped_df, _args(groups=["g"]), _rng())
        row_a = result[result["g"] == "A"]
        assert row_a["correlation"].iloc[0] == pytest.approx(1.0)

    def test_group_b_low_correlation(self, grouped_df):
        result = _corr(grouped_df, _args(groups=["g"]), _rng())
        row_b = result[result["g"] == "B"]
        assert abs(row_b["correlation"].iloc[0]) < 0.7

    def test_multi_group_col(self):
        df = pd.DataFrame(
            {
                "g1": ["X", "X", "Y", "Y"] * 5,
                "g2": ["P", "Q", "P", "Q"] * 5,
                "a": np.arange(20.0),
                "b": np.arange(20.0) * 2,
            }
        )
        result = _corr(df, _args(groups=["g1", "g2"]), _rng())
        assert list(result.columns[:2]) == ["g1", "g2"]
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_p_perm_in_range(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=200), _rng())
        p = result["p_perm"].iloc[0]
        assert 0.0 <= p <= 1.0

    def test_perfectly_correlated_p_perm_small(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=500), _rng(1))
        assert result["p_perm"].iloc[0] < 0.05

    def test_ci_boot_ordered(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=200), _rng())
        lo = result["ci_boot_lo"].iloc[0]
        hi = result["ci_boot_hi"].iloc[0]
        assert lo <= hi

    def test_ci_boot_contains_observed(self, perfect_df):
        result = _corr(perfect_df, _args(bootstrap=500), _rng())
        corr = result["correlation"].iloc[0]
        lo = result["ci_boot_lo"].iloc[0]
        hi = result["ci_boot_hi"].iloc[0]
        assert lo <= corr <= hi

    def test_reproducible_with_seed(self, perfect_df):
        r1 = _corr(perfect_df, _args(bootstrap=100), np.random.default_rng(42))
        r2 = _corr(perfect_df, _args(bootstrap=100), np.random.default_rng(42))
        pd.testing.assert_frame_equal(r1, r2)

    def test_bootstrap_with_groups_one_row_per_group(self, grouped_df):
        result = _corr(grouped_df, _args(bootstrap=100, groups=["g"]), _rng())
        assert len(result) == 2

    def test_spearman_ci_boot(self):
        # Use noisy data — perfectly correlated data is degenerate for BCa
        rng = np.random.default_rng(3)
        x = np.arange(1.0, 31.0)
        df = pd.DataFrame({"a": x, "b": x + rng.normal(0, 2, 30)})
        result = _corr(df, _args(method="spearman", bootstrap=500), _rng())
        assert "ci_boot_lo" in result.columns
        assert result["ci_boot_lo"].iloc[0] <= result["ci_boot_hi"].iloc[0]


# ---------------------------------------------------------------------------
# _bootstrap_stats directly
# ---------------------------------------------------------------------------


class TestBootstrapStats:
    def test_perfect_corr_p_boot_near_zero(self, perfect_df):
        a, b = perfect_df["a"].values, perfect_df["b"].values
        p, lo, hi = _bootstrap_stats(a, b, ss.pearsonr, 500, _rng())
        assert p < 0.05

    def test_p_boot_range(self):
        rng = np.random.default_rng(5)
        a = rng.normal(0, 1, 30)
        b = rng.normal(0, 1, 30)
        p, lo, hi = _bootstrap_stats(a, b, ss.pearsonr, 200, _rng())
        assert 0.0 <= p <= 1.0
        assert lo <= hi


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaN:
    def test_nan_rows_dropped(self):
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})
        result = _corr(df, _args(), _rng())
        assert result["nobs"].iloc[0] == 3

    def test_pvalue_in_range(self, perfect_df):
        for method in METHODS:
            result = _corr(perfect_df, _args(method=method), _rng())
            pval = result["pvalue"].iloc[0]
            assert 0.0 <= pval <= 1.0, f"{method}: pvalue={pval}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_malformed_pair_raises(self, perfect_df):
        with pytest.raises(ValueError, match="col1:col2"):
            _corr(perfect_df, _args(cols=["ab"]), _rng())

    def test_confidence_affects_ci_width(self, perfect_df):
        r95 = _corr(perfect_df, _args(bootstrap=200, confidence=95.0), _rng(1))
        r50 = _corr(perfect_df, _args(bootstrap=200, confidence=50.0), _rng(1))
        width95 = r95["ci_boot_hi"].iloc[0] - r95["ci_boot_lo"].iloc[0]
        width50 = r50["ci_boot_hi"].iloc[0] - r50["ci_boot_lo"].iloc[0]
        assert width95 >= width50
