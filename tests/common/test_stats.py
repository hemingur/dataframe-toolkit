"""
Tests for dftk.common.stats — row-wise statistical functions.
"""

import math

import numpy as np
import pytest
import scipy.stats as ss

from dftk.common.stats import (
    VECTORISED,
    binom_test,
    boschloo_OR,
    boschloo_test,
    chi2_neglogp_array,
    chi2_to_neglogp,
    fisher_OR,
    fisher_test,
    generalized_poisson_nll,
    neglogp_chi2_array,
    neglogp_to_chi2,
    pval2se,
    pval_to_chi2,
    t2pval,
)

# ---------------------------------------------------------------------------
# pval2se
# ---------------------------------------------------------------------------


class TestPval2se:
    def test_known_values(self):
        # effect=1.96, p=0.05 → se ≈ 1.0
        se = pval2se([1.96, 0.05])
        assert se == pytest.approx(1.0, rel=0.01)

    def test_symmetry(self):
        # sign of effect should not matter
        assert pval2se([2.0, 0.01]) == pytest.approx(pval2se([-2.0, 0.01]))


# ---------------------------------------------------------------------------
# t2pval
# ---------------------------------------------------------------------------


class TestT2pval:
    def test_t_zero_gives_pval_one(self):
        assert t2pval([0.0, 10]) == pytest.approx(1.0)

    def test_large_t_gives_small_pval(self):
        assert t2pval([10.0, 100]) < 1e-10

    def test_matches_scipy(self):
        t, df = 2.5, 20
        expected = float(2 * ss.t.sf(abs(t), df))
        assert t2pval([t, df]) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# chi2_to_neglogp / neglogp_to_chi2 round-trip
# ---------------------------------------------------------------------------


class TestChi2NeglogpRoundTrip:
    @pytest.mark.parametrize(
        "chi2,df",
        [
            (3.84, 1),  # classic chi2 at p=0.05
            (10.0, 2),
            (50.0, 5),
            (200.0, 1),  # high chi2 — exercises logsf path
        ],
    )
    def test_round_trip(self, chi2, df):
        neglogp = chi2_to_neglogp([chi2, df])
        recovered = neglogp_to_chi2([neglogp, df])
        assert recovered == pytest.approx(chi2, rel=1e-5)

    def test_moderate_matches_scipy(self):
        chi2, df = 10.0, 3
        expected = float(-ss.chi2.logsf(chi2, df) / np.log(10))
        assert chi2_to_neglogp([chi2, df]) == pytest.approx(expected, rel=1e-8)

    # Exact -log10(p) from mpmath (60 digits); scipy's logsf is -inf for all
    # of these, so they exercise the asymptotic series.
    @pytest.mark.parametrize(
        "chi2,df,expected",
        [
            (1500, 1, 327.4072560441665),
            (1e5, 1, 21717.322159443942),
            (1500, 3, 324.23058649574523),
            (1e5, 3, 21712.322150758228),
            (1500, 50, 280.49799185229256),
            (1e5, 50, 21625.741312221464),
        ],
    )
    def test_past_underflow_matches_exact(self, chi2, df, expected):
        assert chi2_to_neglogp([chi2, df]) == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize("df", [1, 10, 50])
    def test_no_jump_where_scipy_underflows(self, df):
        # The old fixed-threshold version jumped at chi2=1380 (by 0.015 at
        # df=50); the series must continue scipy's values smoothly across
        # the point where logsf becomes -inf.
        x = np.arange(1300.0, 2000.0, 0.5)
        switch = np.isinf(ss.chi2.logsf(x, df))
        assert switch.any() and not switch.all()
        y = chi2_neglogp_array(x, df)
        assert np.all(np.isfinite(y))
        assert np.abs(np.diff(y, 2)).max() < 1e-5

    def test_array_matches_rowwise_and_broadcasts(self):
        chi2 = np.array([0.0, 3.84, 200.0, 2000.0, np.nan, np.inf])
        out = chi2_neglogp_array(chi2, 1)
        assert out[0] == 0.0
        assert out[1] == pytest.approx(-np.log10(0.05), rel=1e-3)
        assert out[3] == pytest.approx(chi2_to_neglogp([2000.0, 1]))
        assert np.isnan(out[4])
        assert np.isposinf(out[5])
        per_df = chi2_neglogp_array(2000.0, np.array([1, 2, 10]))
        assert per_df == pytest.approx(
            [chi2_to_neglogp([2000.0, k]) for k in (1, 2, 10)]
        )

    # Exact chi-squared for a given -log10(p), from mpmath (60 digits).  The
    # old one-term inversion was off by up to 4e-5 (relative) here.
    @pytest.mark.parametrize(
        "neglogp,df,expected",
        [
            (500, 1, 2294.3944153304565),
            (500, 50, 2536.0236833382087),
            (5000, 3, 23035.444223414135),
        ],
    )
    def test_inverse_past_underflow_matches_exact(self, neglogp, df, expected):
        assert neglogp_to_chi2([neglogp, df]) == pytest.approx(expected, rel=1e-13)

    @pytest.mark.parametrize("df", [1, 2, 7, 50, 200])
    def test_round_trip_through_underflow(self, df):
        chi2 = np.array([1.0, 50.0, 1300.0, 1400.0, 1500.0, 3000.0, 1e5])
        # upper tail only: where p rounds to 1, -log10(p) can't be inverted
        chi2 = chi2[ss.chi2.sf(chi2, df) < 0.5]
        back = neglogp_chi2_array(chi2_neglogp_array(chi2, df), df)
        assert back == pytest.approx(chi2, rel=1e-12)

    def test_inverse_nan_and_inf(self):
        out = neglogp_chi2_array([np.nan, np.inf, 0.0], 1)
        assert np.isnan(out[0])
        assert np.isposinf(out[1])
        assert out[2] == 0.0

    def test_pval_to_chi2_inverse_of_chi2_to_neglogp(self):
        # pval_to_chi2 and chi2.isf are direct — just verify consistency
        pval, df = 0.05, 1
        chi2 = pval_to_chi2([pval, df])
        assert chi2 == pytest.approx(ss.chi2.isf(pval, df))


# ---------------------------------------------------------------------------
# Exact tests
# ---------------------------------------------------------------------------


class TestBinomTest:
    def test_fair_coin_centre(self):
        # 50 heads out of 100 — should not be significant
        p = binom_test([50, 100, 0.5])
        assert p > 0.5

    def test_extreme_counts_significant(self):
        # 99 heads out of 100 — highly significant
        p = binom_test([99, 100, 0.5])
        assert p < 1e-20

    def test_matches_scipy(self):
        k, n, prob = 30, 50, 0.5
        expected = ss.binomtest(k, n, prob).pvalue
        assert binom_test([k, n, prob]) == pytest.approx(expected)


class TestFisherTest:
    def test_independent_table(self):
        # perfectly balanced 2×2 table → p should be large
        p = fisher_test([10, 10, 10, 10])
        assert p == pytest.approx(1.0)

    def test_extreme_table_significant(self):
        p = fisher_test([100, 0, 0, 100])
        assert p < 1e-10

    def test_or_greater_than_one(self):
        # more a,d than b,c → OR > 1
        assert fisher_OR([10, 1, 1, 10]) > 1.0

    def test_or_less_than_one(self):
        assert fisher_OR([1, 10, 10, 1]) < 1.0

    def test_matches_scipy(self):
        a, b, c, d = 8, 2, 1, 5
        expected_p = ss.fisher_exact([[a, b], [c, d]])[1]
        expected_or = ss.fisher_exact([[a, b], [c, d]])[0]
        assert fisher_test([a, b, c, d]) == pytest.approx(expected_p)
        assert fisher_OR([a, b, c, d]) == pytest.approx(expected_or)


class TestBoschloo:
    def test_independent_not_significant(self):
        p = boschloo_test([5, 5, 5, 5])
        assert p > 0.05

    def test_extreme_significant(self):
        p = boschloo_test([20, 0, 0, 20])
        assert p < 1e-5

    def test_or_symmetric_table(self):
        # a=d, b=c → OR = (ad)/(bc) = 1
        assert boschloo_OR([5, 5, 5, 5]) == pytest.approx(1.0)

    def test_or_unbalanced(self):
        assert boschloo_OR([10, 2, 2, 10]) > 1.0

    def test_or_zero_denominator(self):
        # b=0, c=0 → OR = inf
        result = boschloo_OR([5, 0, 0, 5])
        assert math.isinf(result)


# ---------------------------------------------------------------------------
# generalized_poisson_nll
# ---------------------------------------------------------------------------


class TestGeneralizedPoissonNll:
    def test_positive_nll(self):
        # NLL should be positive (it's -log P, and P < 1)
        result = generalized_poisson_nll([3, 2.0, 0.5])
        assert result > 0

    def test_invalid_params_returns_inf(self):
        # lambda + theta*x <= 0 → undefined → inf
        result = generalized_poisson_nll([10, 1.0, -1.0])
        assert math.isinf(result)

    def test_zero_count(self):
        # x=0 is valid for GP when lambda > 0, theta >= 0
        result = generalized_poisson_nll([0, 1.0, 0.0])
        assert math.isfinite(result)
        assert result > 0


# ---------------------------------------------------------------------------
# Whole-column versions agree with the row functions
# ---------------------------------------------------------------------------


class TestVectorised:
    @pytest.mark.parametrize(
        "name,columns",
        [
            ("pval2se", ([0.1, -0.5, 2.0], [0.05, 1e-10, 0.9])),
            ("t2pval", ([2.0, -3.5, 0.0], [10, 3, 100])),
            ("chi2_to_neglogp", ([0.5, 30.0, 2500.0], [1, 4, 2])),
            ("neglogp_to_chi2", ([1.3, 50.0, 800.0], [1, 4, 2])),
            ("pval_to_chi2", ([0.05, 1e-12, 0.5], [1, 3, 2])),
        ],
    )
    def test_matches_row_function(self, name, columns):
        import dftk.common.stats as stats

        row_func = getattr(stats, name)
        expected = [row_func(list(row)) for row in zip(*columns, strict=True)]
        got = VECTORISED[name](*(np.array(c, dtype=float) for c in columns))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_scalar_broadcasts(self):
        got = VECTORISED["t2pval"](np.array([2.0, 3.0]), 10)
        assert got == pytest.approx([t2pval([2.0, 10]), t2pval([3.0, 10])])
