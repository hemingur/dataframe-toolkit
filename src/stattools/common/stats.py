"""
stattools.common.stats — reusable row-wise statistical functions.

All functions here take a row (array-like indexed by column name) and return
a scalar.  They are intended to be used as pandas .apply(func, axis=1) targets
via the dfstat eval command, but are importable by any other command.

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


def pval2se(row) -> float:
    """Convert an effect estimate and its p-value to a standard error.

    row[0]: effect estimate (beta / log-OR / etc.)
    row[1]: two-sided p-value
    """
    return float(np.abs(row[0] / ss.norm.ppf(row[1] / 2)))


def t2pval(row) -> float:
    """Two-sided p-value from a t-statistic and degrees of freedom.

    row[0]: t-statistic
    row[1]: degrees of freedom
    """
    return float(2 * ss.t.sf(np.abs(row[0]), row[1]))


def chi2_to_neglogp(row, threshold: float = 1380) -> float:
    """Convert a chi-squared statistic and df to -log10(p).

    Uses scipy.stats.chi2.logsf for moderate values; switches to an
    asymptotic approximation above *threshold* to avoid underflow.

    row[0]: chi-squared statistic
    row[1]: degrees of freedom
    """
    chi_val = float(row[0])
    df = float(row[1])
    if chi_val < threshold:
        log_p = ss.chi2.logsf(chi_val, df)
        return float(-log_p / np.log(10))
    else:
        lin_term = chi_val / 2.0
        log_term = -(df / 2 - 1) * np.log(chi_val / 2)
        constant = gammaln(df / 2)
        return float((lin_term + log_term + constant) / np.log(10))


def neglogp_to_chi2(
    row, threshold: float = 300, max_iter: int = 50, tol: float = 1e-8
) -> float:
    """Convert -log10(p) and df to a chi-squared statistic.

    Uses scipy.stats.chi2.isf for moderate -log10(p) values; switches to a
    Newton-Raphson inversion of the asymptotic tail formula above *threshold*.
    Closed-form approximation is used for df=2.

    row[0]: -log10(p)
    row[1]: degrees of freedom
    """
    neglogp = float(row[0])
    df = int(row[1])

    if neglogp < threshold:
        p = 10 ** (-neglogp)
        return float(ss.chi2.isf(p, df))

    constant = gammaln(df / 2)
    nlp = neglogp * np.log(10)
    a = df / 2 - 1
    x = 2 * (nlp - constant)
    if df == 2:
        return float(x)
    for _ in range(max_iter):
        f = nlp - constant + a * np.log(x / 2) - x / 2
        fprime = a / x - 0.5
        step = f / fprime
        x_new = x - step
        if abs(step) < tol:
            return float(x_new)
        x = x_new
    return float(x)


def pval_to_chi2(row) -> float:
    """Convert a p-value and df to a chi-squared statistic.

    row[0]: p-value
    row[1]: degrees of freedom
    """
    return float(ss.chi2.isf(float(row[0]), df=int(row[1])))


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
