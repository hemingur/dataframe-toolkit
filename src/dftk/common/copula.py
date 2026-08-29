"""
dftk.common.copula — nonparametric copula diagnostics: tail dependence and
the corner-resolved chi-plot.

These are dependence-shape diagnostics that sit alongside dftk.corr's
Pearson/Spearman/Kendall coefficients: a correlation coefficient describes
overall association, but says nothing about whether the *extremes* of two
variables move together more (or less) than the bulk of the data does. Two
column pairs can have the same Spearman rho and very different tail
behavior -- this module answers that separately. Domain-agnostic: the same
diagnostics apply whether the two columns are genetic markers, asset
returns, or climate variables.

Public API
----------
pseudo_observations(x, y)          Rank-transform two raw arrays to
                                    approximately Uniform(0,1) margins.
bivariate_loo_cdf(u, v)            O(n log n) leave-one-out bivariate
                                    empirical CDF (Fenwick tree), tie-safe.
empirical_tail_dependence(u, v, q_grid)
                                    Nonparametric lower/upper tail-dependence
                                    coefficient curves.
corner_chi(u, v)                   Corner-resolved chi statistic and x-axis
                                    coordinate for the chi-plot (dftk chi).

All functions expect pseudo-observations (u, v already close to Uniform(0,1)
margins) except pseudo_observations itself, which produces them from raw
data via rank / (n + 1).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

__all__ = [
    "pseudo_observations",
    "bivariate_loo_cdf",
    "empirical_tail_dependence",
    "corner_chi",
    "MIN_TAIL_N",
]

# Below this many points landing in the tail region, a tail-dependence
# estimate at a given q is dominated by sampling noise rather than signal.
# Callers should warn (not silently trust) when n * q drops below this.
MIN_TAIL_N = 20


def pseudo_observations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-transform two raw arrays to approximately Uniform(0,1) margins.

    This is the step that isolates dependence structure from marginal shape:
    each variable is replaced by its own empirical CDF value, so what remains
    describes only how the two variables move together. x and y must already
    be aligned (same length, paired observations, no NaNs -- callers should
    do pairwise-deletion before calling this, matching the corr command's
    own convention).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    u = rankdata(x, method="average") / (n + 1)
    v = rankdata(y, method="average") / (n + 1)
    return u, v


def bivariate_loo_cdf(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """O(n log n) leave-one-out bivariate empirical CDF via a Fenwick tree:

        H_i = #{j != i : u_j <= u_i and v_j <= v_i} / (n - 1)

    A naive O(n^2) version (looping over i with a vectorized comparison
    against all j) is fine for a few thousand rows, but real-world tables
    (genomic windows, tick data, climate grids) commonly run into the
    hundreds of thousands of rows, where an O(n^2) loop stalls for minutes;
    this scales to those sizes in about a second.

    Handles ties in u and/or v exactly, including fully duplicate (u, v)
    rows -- real data (rounded measurements, binned values, repeated
    readings) ties often enough that this can't be waved away. Points are
    processed in ascending-u batches: each batch first queries the Fenwick
    tree (built from strictly-smaller-u points only), then resolves mutual
    counts *within* the batch by the same leave-one-out logic applied to v,
    before finally inserting the batch into the tree.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    n = len(u)
    if n < 2:
        return np.zeros(n)

    v_unique, v_idx = np.unique(v, return_inverse=True)
    m = len(v_unique)
    tree = np.zeros(m + 1, dtype=np.int64)

    def update(i: int) -> None:
        i += 1
        while i <= m:
            tree[i] += 1
            i += i & (-i)

    def query(i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    order = np.argsort(u, kind="mergesort")
    u_sorted = u[order]

    H = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and u_sorted[j + 1] == u_sorted[i]:
            j += 1
        batch = order[i : j + 1]

        # Contribution from points with strictly smaller u, already in the
        # tree.
        ext = np.array([query(int(v_idx[k])) for k in batch])

        # Contribution from other points in this same u-tie batch: a smaller
        # instance of the identical leave-one-out counting problem, now
        # over v alone since u is constant across the batch.
        batch_v = v[batch]
        b_order = np.argsort(batch_v, kind="mergesort")
        b_sorted = batch_v[b_order]
        within = np.zeros(len(batch), dtype=np.int64)
        bi, cum = 0, 0
        while bi < len(batch):
            bj = bi
            while bj + 1 < len(batch) and b_sorted[bj + 1] == b_sorted[bi]:
                bj += 1
            tie_size = bj - bi + 1
            # Every point in this v-tie sub-group: cum (strictly smaller v
            # within the batch) plus the other tie_size - 1 members tied
            # with it on both u and v (mutual, excluding self).
            within[b_order[bi : bj + 1]] = cum + tie_size - 1
            cum += tie_size
            bi = bj + 1

        H[batch] = ext + within

        for k in batch:
            update(int(v_idx[k]))
        i = j + 1

    return H / (n - 1)


def empirical_tail_dependence(
    u: np.ndarray, v: np.ndarray, q_grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nonparametric lower- and upper-tail dependence coefficient curves.

        lambda_L(q) = C_n(q, q) / q                    -- lower tail, q -> 0
        lambda_U(q) = (1 - 2(1-q) + C_n(1-q, 1-q)) / q  -- upper tail, q -> 0

    where C_n is the empirical copula. Both estimate the true coefficient in
    the limit q -> 0. IMPORTANT: at any *finite* q these are not flat at the
    true value -- even a copula with zero tail dependence gives a nonzero,
    q-dependent reading here, growing with the strength of overall
    correlation, and only decays toward 0 as q shrinks. A copula with
    genuine tail dependence instead levels off at a stable nonzero value.
    That decay-vs-plateau distinction across q, not the reading at a single
    q, is the signal -- see MIN_TAIL_N for when a given q is trustworthy at
    a given sample size.
    """
    lam_L = np.empty(len(q_grid))
    lam_U = np.empty(len(q_grid))
    for i, q in enumerate(q_grid):
        C_L = np.mean((u <= q) & (v <= q))
        lam_L[i] = C_L / q if q > 0 else np.nan
        uq = 1 - q
        C_U = np.mean((u <= uq) & (v <= uq))
        lam_U[i] = (1 - 2 * uq + C_U) / q if q > 0 else np.nan
    return lam_L, lam_U


def corner_chi(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corner-resolved chi statistic, based on Fisher & Switzer (1985, 2001).

    Returns (x, chi) for concordant pairs only, where x in (-1, 1):
    x -> -1 is deep in the lower-left corner (both variables jointly small),
    x -> +1 is deep in the upper-right corner (both jointly large).

    IMPORTANT: the textbook chi-plot's sign convention splits pairs into
    "concordant" (both above or both below their median -- i.e. EITHER
    corner) vs "discordant", which cannot by itself distinguish lower-tail
    dependence (e.g. Clayton copula) from upper-tail dependence (e.g. Gumbel
    copula) -- both corners land on the same side of its x-axis. This
    version restricts to concordant pairs and signs x by which corner each
    point is actually in, so the two corners are genuinely separated.

    Reads as: chi elevated (and rising) on the negative side only => lower-
    tail dependence. Elevated on the positive side only => upper-tail
    dependence. Elevated similarly on both sides, decaying well before the
    edges => generic correlation with no real tail dependence. Flat at 0
    everywhere => independence.
    """
    F, G = u, v  # already pseudo-observations, which lie strictly in
    # (0, 1) -- never exactly 0 or 1 -- so F*(1-F) and G*(1-G) below never
    # hit zero and the chi formula can't divide by zero.
    H = bivariate_loo_cdf(u, v)
    with np.errstate(invalid="ignore", divide="ignore"):
        chi = (H - F * G) / np.sqrt(F * (1 - F) * G * (1 - G))
    concordant = np.sign(F - 0.5) == np.sign(G - 0.5)
    x = np.sign(F - 0.5) * 4 * np.maximum((F - 0.5) ** 2, (G - 0.5) ** 2)
    return x[concordant], chi[concordant]
