"""
dftk.common.stats — reusable statistical functions.

The functions listed below take a row (array-like, values in argument order)
and return a scalar, for eval -f.  The p-value / chi-squared conversions also
have whole-column versions (``*_array``, collected in VECTORISED) that take
arrays and broadcast; eval uses those when they exist.

Available functions
-------------------
pval2se(row)            Convert p-value and effect estimate to standard error.
t2pval(row)             Two-sided p-value from t-statistic and degrees of freedom.
chi2_to_neglogp(row)    Chi-squared + df  →  -log10(p), numerically stable.
neglogp_to_chi2(row)    -log10(p) + df  →  chi-squared, numerically stable.
pval_to_chi2(row)       p-value + df  →  chi-squared.
binom_test(row)         Binomial test p-value.
fisher_test(row)        Fisher exact test p-value (2×2 table).
fisher_OR(row)          Fisher exact test odds ratio (2×2 table).
boschloo_test(row)      Boschloo exact test p-value (2×2 table).
boschloo_OR(row)        Odds ratio from Boschloo test (2×2 table).
generalized_poisson_nll(row)  Negative log-likelihood for the Generalised Poisson.
"""

import numpy as np
import scipy.stats as ss
from scipy.special import gammaln

__all__ = [
    "pval2se",
    "t2pval",
    "chi2_to_neglogp",
    "chi2_neglogp_array",
    "neglogp_chi2_array",
    "pval_chi2_array",
    "pval2se_array",
    "t2pval_array",
    "VECTORISED",
    "neglogp_to_chi2",
    "pval_to_chi2",
    "binom_test",
    "fisher_test",
    "fisher_OR",
    "boschloo_test",
    "boschloo_OR",
    "generalized_poisson_nll",
]

# ---------------------------------------------------------------------------
# p-value / chi-squared conversions
# ---------------------------------------------------------------------------


def _floats(*args) -> list[np.ndarray]:
    """Broadcast scalars / arrays / Series to float arrays of one shape."""
    return np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in args))


def pval2se_array(effect, pval) -> np.ndarray:
    """Vectorised :func:`pval2se`: |effect / z| with z = Phi^-1(p / 2)."""
    effect, pval = _floats(effect, pval)
    return np.abs(effect / ss.norm.ppf(pval / 2))


def pval2se(row) -> float:
    """Convert an effect estimate and its p-value to a standard error.

    row[0]: effect estimate (beta / log-OR / etc.)
    row[1]: two-sided p-value
    """
    return float(pval2se_array(row[0], row[1]))


def t2pval_array(t, df) -> np.ndarray:
    """Vectorised :func:`t2pval`."""
    t, df = _floats(t, df)
    return 2 * ss.t.sf(np.abs(t), df)


def t2pval(row) -> float:
    """Two-sided p-value from a t-statistic and degrees of freedom.

    row[0]: t-statistic
    row[1]: degrees of freedom
    """
    return float(t2pval_array(row[0], row[1]))


def chi2_neglogp_array(chi2, df) -> np.ndarray:
    """Vectorised -log10(p) for chi-squared statistics, exact past underflow.

    scipy's chi2.logsf is exact for as long as it is finite (p down to about
    1e-308); beyond that it returns -inf, and this function uses the
    asymptotic expansion of the upper incomplete gamma function instead:

        Q(s, z) ~ z^(s-1) e^(-z) / Gamma(s) * sum_n (s-1)(s-2)...(s-n) / z^n

    with s = df/2, z = chi2/2.  The series is cut at its smallest term (it is
    asymptotic, so terms eventually grow); for integer s it terminates and is
    exact.  Checked against mpmath to ~1e-15 for df 1..1000 (tests/common).

    chi2, df: scalars or arrays (broadcast).  NaN in gives NaN out.
    """
    chi2, df = _floats(chi2, df)
    with np.errstate(divide="ignore"):
        out = np.array(-ss.chi2.logsf(chi2, df) / np.log(10) + 0.0, dtype=float)
    under = np.isposinf(out) & np.isfinite(chi2)
    if under.any():
        s, z = df[under] / 2, chi2[under] / 2
        series = np.ones_like(z)
        term = np.ones_like(z)
        active = np.ones(z.shape, dtype=bool)
        for n in range(1, 40):
            nxt = term * (s - n) / z
            active &= np.abs(nxt) < np.abs(term)
            if not active.any():
                break
            term = np.where(active, nxt, term)
            series = np.where(active, series + nxt, series)
        neg_ln_p = z - (s - 1) * np.log(z) + gammaln(s) - np.log(series)
        out[under] = neg_ln_p / np.log(10)
    return out


def chi2_to_neglogp(row) -> float:
    """Convert a chi-squared statistic and df to -log10(p).

    Row-wise wrapper around :func:`chi2_neglogp_array`.

    row[0]: chi-squared statistic
    row[1]: degrees of freedom
    """
    return float(chi2_neglogp_array(row[0], row[1]))


def neglogp_chi2_array(neglogp, df, threshold: float = 300) -> np.ndarray:
    """Vectorised inverse of :func:`chi2_neglogp_array`: -log10(p), df → chi2.

    Below *threshold*, scipy's chi2.isf(10^-neglogp).  Above it (p near or
    past underflow), Newton's method on chi2_neglogp_array, started from the
    one-term asymptotic inverse, so the two functions round-trip exactly.
    df is truncated to an integer, as it always was.
    """
    neglogp, df = _floats(neglogp, df)
    df = np.trunc(df)
    out = np.full(neglogp.shape, np.nan)
    low = neglogp < threshold
    out[low] = ss.chi2.isf(10 ** (-neglogp[low]), df[low])
    high = (neglogp >= threshold) & np.isfinite(neglogp) & (df > 0)
    if high.any():
        target, k = neglogp[high], df[high]
        s = k / 2
        nlp = target * np.log(10)
        # start: one-term asymptotic inverse, by fixed-point iteration
        x = 2 * (nlp - gammaln(s))
        for _ in range(20):
            x = 2 * (nlp - gammaln(s) + (s - 1) * np.log(x / 2))
        for _ in range(50):
            f = chi2_neglogp_array(x, k) - target
            slope = (0.5 - (s - 1) / x) / np.log(10)  # d(-log10 p)/dx, asymptotic
            step = f / slope
            x = x - step
            if np.all(np.abs(step) <= 1e-13 * x):
                break
        out[high] = x
    out[np.isposinf(neglogp)] = np.inf
    return out


def neglogp_to_chi2(row) -> float:
    """Convert -log10(p) and df to a chi-squared statistic.

    Row-wise wrapper around :func:`neglogp_chi2_array`.

    row[0]: -log10(p)
    row[1]: degrees of freedom
    """
    return float(neglogp_chi2_array(row[0], row[1]))


def pval_chi2_array(pval, df) -> np.ndarray:
    """Vectorised :func:`pval_to_chi2` (df truncated to an integer)."""
    pval, df = _floats(pval, df)
    return ss.chi2.isf(pval, np.trunc(df))


def pval_to_chi2(row) -> float:
    """Convert a p-value and df to a chi-squared statistic.

    row[0]: p-value
    row[1]: degrees of freedom
    """
    return float(pval_chi2_array(row[0], row[1]))


# Whole-column versions of the row functions above, for eval -f: same
# arguments in the same order, numpy arrays (or scalars) instead of a row.
VECTORISED = {
    "pval2se": pval2se_array,
    "t2pval": t2pval_array,
    "chi2_to_neglogp": chi2_neglogp_array,
    "neglogp_to_chi2": neglogp_chi2_array,
    "pval_to_chi2": pval_chi2_array,
}


# ---------------------------------------------------------------------------
# Exact tests
# ---------------------------------------------------------------------------


def binom_test(row) -> float:
    """Binomial test p-value.

    row[0]: number of successes (int)
    row[1]: number of trials (int)
    row[2]: hypothesised success probability
    """
    return float(ss.binomtest(int(row[0]), int(row[1]), float(row[2])).pvalue)


def fisher_test(row) -> float:
    """Fisher exact test p-value for a 2×2 contingency table.

    row[0..3]: a, b, c, d  (cells of the table, row-major)
    """
    return float(ss.fisher_exact([[row[0], row[1]], [row[2], row[3]]])[1])


def fisher_OR(row) -> float:
    """Fisher exact test odds ratio for a 2×2 contingency table.

    row[0..3]: a, b, c, d  (cells of the table, row-major)
    """
    return float(ss.fisher_exact([[row[0], row[1]], [row[2], row[3]]])[0])


def boschloo_test(row) -> float:
    """Boschloo exact test p-value for a 2×2 contingency table.

    Boschloo's test is uniformly more powerful than Fisher's exact test for
    testing independence when both margins are fixed.

    row[0..3]: a, b, c, d  (cells of the table, row-major)
    """
    table = [[int(row[0]), int(row[1])], [int(row[2]), int(row[3])]]
    return float(ss.boschloo_exact(table).pvalue)


def boschloo_OR(row) -> float:
    """Odds ratio from a 2×2 contingency table (same as Fisher OR).

    Boschloo's test does not produce an OR directly; we return the
    standard maximum-likelihood OR (ad/bc) instead.

    row[0..3]: a, b, c, d  (cells of the table, row-major)
    """
    a, b, c, d = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    denom = b * c
    if denom == 0:
        return float("inf") if a * d > 0 else float("nan")
    return float((a * d) / denom)


# ---------------------------------------------------------------------------
# Probability distributions
# ---------------------------------------------------------------------------


def generalized_poisson_nll(row) -> float:
    """Negative log-likelihood for the Generalised Poisson distribution.

    Vectorised-safe: operates on scalar values extracted from a row.

    row[0]: observed count x
    row[1]: lambda parameter
    row[2]: theta parameter

    Returns -log P(x | lambda, theta), or +inf when lambda + theta*x ≤ 0.

    log P = log(lambda) + (x-1)*log(lambda + theta*x)
            - (lambda + theta*x) - log(x!)
    """
    x = float(row[0])
    lam = float(row[1])
    theta = float(row[2])
    lam_theta_x = lam + theta * x
    if lam_theta_x <= 0 or lam <= 0:
        return float("inf")
    log_pmf = np.log(lam) + (x - 1) * np.log(lam_theta_x) - lam_theta_x - gammaln(x + 1)
    return float(-log_pmf)
