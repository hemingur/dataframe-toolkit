"""
Tests for stattools.common.stats — row-wise statistical functions.
"""

import math

import numpy as np
import pytest
import scipy.stats as ss

from stattools.common.stats import (
    pval2se,
    t2pval,
    chi2_to_neglogp,
    neglogp_to_chi2,
    pval_to_chi2,
    binom_test,
    fisher_test,
    fisher_OR,
    boschloo_test,
    boschloo_OR,
    generalized_poisson_nll,
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

    @pytest.mark.parametrize("chi2,df", [
        (3.84, 1),   # classic chi2 at p=0.05
        (10.0, 2),
        (50.0, 5),
        (200.0, 1),  # high chi2 — exercises logsf path
    ])
    def test_round_trip(self, chi2, df):
        neglogp = chi2_to_neglogp([chi2, df])
        recovered = neglogp_to_chi2([neglogp, df])
        assert recovered == pytest.approx(chi2, rel=1e-5)

    def test_moderate_matches_scipy(self):
        chi2, df = 10.0, 3
        expected = float(-ss.chi2.logsf(chi2, df) / np.log(10))
        assert chi2_to_neglogp([chi2, df]) == pytest.approx(expected, rel=1e-8)

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
        expected_p  = ss.fisher_exact([[a, b], [c, d]])[1]
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
