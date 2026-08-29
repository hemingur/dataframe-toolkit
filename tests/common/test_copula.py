"""
Tests for dftk.common.copula — nonparametric tail-dependence diagnostics.
"""

import numpy as np
import pytest
from scipy.stats import rankdata

from dftk.common.copula import (
    MIN_TAIL_N,
    bivariate_loo_cdf,
    corner_chi,
    empirical_tail_dependence,
    pseudo_observations,
)

# ---------------------------------------------------------------------------
# Reference implementations (naive O(n^2), used to validate the fast paths)
# ---------------------------------------------------------------------------


def _naive_loo_cdf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    n = len(u)
    H = np.zeros(n)
    for i in range(n):
        H[i] = np.sum((u <= u[i]) & (v <= v[i])) - 1
    return H / (n - 1)


def _clayton_sample(theta: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Draw (u, v) directly on Uniform(0,1) margins from a Clayton copula.

    True tail dependence: lambda_lower = 2^(-1/theta), lambda_upper = 0.
    """
    rng = np.random.default_rng(seed)
    w = rng.random(n)
    t = rng.random(n)
    u = w
    v = (w ** (-theta) * (t ** (-theta / (theta + 1)) - 1) + 1) ** (-1 / theta)
    return u, v


# ---------------------------------------------------------------------------
# pseudo_observations
# ---------------------------------------------------------------------------


class TestPseudoObservations:
    def test_matches_scipy_rankdata(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 10, size=50).astype(float)  # heavy ties
        y = rng.random(50)
        u, v = pseudo_observations(x, y)
        n = len(x)
        assert np.allclose(u, rankdata(x, method="average") / (n + 1))
        assert np.allclose(v, rankdata(y, method="average") / (n + 1))

    def test_range_strictly_inside_unit_interval(self):
        rng = np.random.default_rng(1)
        x = rng.random(30)
        y = rng.random(30)
        u, v = pseudo_observations(x, y)
        assert u.min() > 0.0
        assert u.max() < 1.0
        assert v.min() > 0.0
        assert v.max() < 1.0

    def test_monotone_in_input_order(self):
        x = np.array([3.0, 1.0, 2.0])
        y = np.array([30.0, 10.0, 20.0])
        u, v = pseudo_observations(x, y)
        assert u[1] < u[2] < u[0]
        assert v[1] < v[2] < v[0]


# ---------------------------------------------------------------------------
# bivariate_loo_cdf — correctness against a naive O(n^2) reference
# ---------------------------------------------------------------------------


class TestBivariateLooCdf:
    def test_no_ties(self):
        rng = np.random.default_rng(42)
        u, v = rng.random(30), rng.random(30)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_ties_in_u_only(self):
        rng = np.random.default_rng(42)
        u = rng.integers(0, 8, 40).astype(float)
        v = rng.random(40)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_ties_in_v_only(self):
        rng = np.random.default_rng(42)
        u = rng.random(40)
        v = rng.integers(0, 8, 40).astype(float)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_ties_in_both_u_and_v(self):
        rng = np.random.default_rng(42)
        u = rng.integers(0, 6, 50).astype(float)
        v = rng.integers(0, 6, 50).astype(float)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_duplicate_rows(self):
        rng = np.random.default_rng(42)
        base_u, base_v = rng.random(10), rng.random(10)
        u = np.concatenate([base_u, base_u, base_u])
        v = np.concatenate([base_v, base_v, base_v])
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_all_points_identical(self):
        u, v = np.ones(20), np.ones(20)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_realistic_rounded_data(self):
        # Mimics heavily-rounded real columns (e.g. carat: rounded to 2dp),
        # which is exactly the shape that breaks tie-unsafe implementations.
        rng = np.random.default_rng(7)
        x = np.round(rng.gamma(2, 0.3, size=800), 2)
        y = x * 3000 + rng.normal(0, 500, 800)
        u, v = pseudo_observations(x, y)
        assert np.allclose(bivariate_loo_cdf(u, v), _naive_loo_cdf(u, v))

    def test_single_point(self):
        assert bivariate_loo_cdf(np.array([0.5]), np.array([0.5])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# empirical_tail_dependence
# ---------------------------------------------------------------------------


class TestEmpiricalTailDependence:
    def test_clayton_known_lower_tail_dependence(self):
        theta = 3.0
        u, v = _clayton_sample(theta, n=20000, seed=123)
        lam_L, lam_U = empirical_tail_dependence(u, v, np.array([0.01, 0.02, 0.05]))
        true_lambda_L = 2 ** (-1 / theta)
        assert lam_L == pytest.approx(true_lambda_L, abs=0.1)
        assert np.all(lam_U < 0.2)  # Clayton has zero upper tail dependence

    def test_independence_gives_low_lambda(self):
        rng = np.random.default_rng(0)
        u, v = rng.random(5000), rng.random(5000)
        lam_L, lam_U = empirical_tail_dependence(u, v, np.array([0.05]))
        assert lam_L[0] < 0.2
        assert lam_U[0] < 0.2

    def test_q_zero_is_nan(self):
        u, v = np.array([0.1, 0.5, 0.9]), np.array([0.1, 0.5, 0.9])
        lam_L, lam_U = empirical_tail_dependence(u, v, np.array([0.0]))
        assert np.isnan(lam_L[0])
        assert np.isnan(lam_U[0])

    def test_output_shape_matches_q_grid(self):
        rng = np.random.default_rng(0)
        u, v = rng.random(100), rng.random(100)
        q_grid = np.array([0.05, 0.1, 0.2])
        lam_L, lam_U = empirical_tail_dependence(u, v, q_grid)
        assert len(lam_L) == len(q_grid)
        assert len(lam_U) == len(q_grid)


# ---------------------------------------------------------------------------
# corner_chi
# ---------------------------------------------------------------------------


class TestCornerChi:
    def test_x_range(self):
        rng = np.random.default_rng(0)
        u, v = rng.random(500), rng.random(500)
        x, chi = corner_chi(u, v)
        assert x.min() >= -1.0
        assert x.max() <= 1.0

    def test_lower_corner_points_negative_x(self):
        u = np.array([0.05, 0.9])
        v = np.array([0.05, 0.9])
        x, chi = corner_chi(u, v)
        assert x[0] < 0
        assert x[1] > 0

    def test_clayton_asymmetric_chi(self):
        # Clayton has lower-tail dependence only: chi in the deep lower
        # corner should be elevated relative to the deep upper corner.
        u, v = _clayton_sample(theta=4.0, n=5000, seed=1)
        x, chi = corner_chi(u, v)
        lower_deep = chi[(x > -0.9) & (x < -0.6)]
        upper_deep = chi[(x > 0.6) & (x < 0.9)]
        assert lower_deep.mean() > upper_deep.mean()

    def test_no_division_by_zero_warnings(self):
        rng = np.random.default_rng(0)
        u, v = rng.random(200), rng.random(200)
        with np.errstate(all="raise"):
            corner_chi(u, v)

    def test_ties_do_not_crash(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 5, 200).astype(float)
        y = rng.integers(0, 5, 200).astype(float)
        u, v = pseudo_observations(x, y)
        x_chi, chi = corner_chi(u, v)
        assert np.all(np.isfinite(chi))


def test_min_tail_n_is_positive():
    assert MIN_TAIL_N > 0
