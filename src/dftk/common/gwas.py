"""
dftk.common.gwas — shared helpers for genome-wide plots (manhattan, locus).

Column lookup (defaults Chr / pos / Pval, with common spellings as a
fallback), -log10(p) from a p-value, an already-transformed -log10(p) or a
chi-squared column, chromosome order, genome-wide x positions, and thinning
of the many uninteresting points of a large association scan.

Only numpy and pandas at module level: scipy is imported inside
``neglogp_from_chi2`` (via ``common.stats``), the one place that needs it.
"""

from __future__ import annotations

import logging
import math
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CHR = "Chr"
DEFAULT_POS = "pos"
DEFAULT_P = "Pval"

# Spellings tried (ignoring case) when the default column name is missing.
CHR_ALIASES = ("chr", "chrom", "#chrom", "#chr", "chromosome")
POS_ALIASES = ("pos", "bp", "position", "base_pair_location", "genpos")
P_ALIASES = ("pval", "p", "p_value", "pvalue", "p-value", "p.value")
MLOGP_ALIASES = ("log10p", "mlog10p", "neglog10p", "neg_log10_p", "-log10p")
CHI2_ALIASES = ("chisq", "chi2", "chi_sq", "chisquare")

# -log10 of the smallest positive double (denormal 5e-324): a p-value of 0
# is "at least this significant", and the plot shows it at this height.
MAX_NEGLOGP_FROM_P = -math.log10(5e-324)


# ---------------------------------------------------------------------------
# Column lookup
# ---------------------------------------------------------------------------


def _alias_matches(df: pd.DataFrame, aliases: tuple[str, ...]) -> list[str]:
    wanted = {a.casefold() for a in aliases}
    return [c for c in df.columns if str(c).casefold() in wanted]


def find_column(
    df: pd.DataFrame,
    given: str | None,
    default: str,
    aliases: tuple[str, ...],
    option: str,
) -> str:
    """Return the column to use for one role (chromosome, position, ...).

    *given* (the user's option) must exist.  Without it, *default* is used if
    present, else the one column matching *aliases* ignoring case.  No match,
    or several, is an error that names *option*.
    """
    if given is not None:
        if given not in df.columns:
            raise ValueError(
                f"Column {given!r} ({option}) not found.  "
                f"Available columns: {list(df.columns)}"
            )
        return given
    if default in df.columns:
        return default
    found = _alias_matches(df, aliases + (default,))
    if len(found) == 1:
        logger.info("Using column %r for %s", found[0], option)
        return found[0]
    if found:
        raise ValueError(
            f"Several columns could be {option}: {found}; choose one with {option}."
        )
    raise ValueError(
        f"No column {default!r} (or {', '.join(aliases)}, ignoring case); "
        f"name it with {option}.  Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# -log10(p)
# ---------------------------------------------------------------------------


def neglogp_from_p(p: pd.Series | np.ndarray) -> np.ndarray:
    """-log10(p); p = 0 (underflow) is capped at MAX_NEGLOGP_FROM_P with a warning.

    p outside [0, 1] is an error.  NaN stays NaN.
    """
    p = np.asarray(p, dtype=float)
    bad = (p < 0) | (p > 1)
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} p-values are outside [0, 1] "
            f"(e.g. {p[bad][0]!r}); is this a p-value column?"
        )
    zero = p == 0
    with np.errstate(divide="ignore"):
        out = -np.log10(p) + 0.0
    if zero.any():
        out[zero] = MAX_NEGLOGP_FROM_P
        logger.warning(
            "%d p-values are 0 (below about 1e-308, the smallest a number can "
            "hold); drawn at -log10(p) = %.1f.  A chi-squared column "
            "(--chi2col) gives their true size.",
            int(zero.sum()),
            MAX_NEGLOGP_FROM_P,
        )
    return out


def check_neglogp(values: pd.Series | np.ndarray, col: str) -> np.ndarray:
    """Validate an already -log10(p) column: mostly negative means log10(p)."""
    v = np.asarray(values, dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size and (finite < 0).mean() > 0.5:
        raise ValueError(
            f"Column {col!r} is mostly negative, so it looks like log10(p) "
            f"rather than -log10(p).  Negate it first: "
            f"dftk eval -f 'mlogp = -{col}' … --mlogpcol mlogp"
        )
    return v


def neglogp_from_chi2(chi2: pd.Series | np.ndarray, df) -> np.ndarray:
    """-log10(p) from chi-squared statistics; exact far past p = 1e-308."""
    from dftk.common.stats import chi2_neglogp_array

    return chi2_neglogp_array(np.asarray(chi2, dtype=float), df)


def resolve_neglogp(
    data: pd.DataFrame,
    pcol: str | None = None,
    mlogpcol: str | None = None,
    chi2col: str | None = None,
    chi2df: str | float = 1,
) -> tuple[np.ndarray, str]:
    """Return (-log10(p) values, source column) from whichever column is given.

    At most one of *pcol*, *mlogpcol* and *chi2col* may be given.  With none,
    look for ``Pval`` (or a p-value spelling), then a -log10(p) spelling
    (regenie's ``LOG10P``), then a chi-squared one.  *chi2df* is a number or
    the name of a column holding each row's degrees of freedom.
    """
    given = [o for o in (pcol, mlogpcol, chi2col) if o is not None]
    if len(given) > 1:
        raise ValueError("Give only one of --pcol, --mlogpcol and --chi2col.")

    if mlogpcol is None and chi2col is None:
        if pcol is not None or DEFAULT_P in data.columns:
            pcol = find_column(data, pcol, DEFAULT_P, P_ALIASES, "--pcol")
        else:
            for aliases, kind in (
                (P_ALIASES, "p"),
                (MLOGP_ALIASES, "mlogp"),
                (CHI2_ALIASES, "chi2"),
            ):
                found = _alias_matches(data, aliases)
                if len(found) > 1:
                    raise ValueError(
                        f"Several columns could hold the p-values: {found}; "
                        "choose one with --pcol, --mlogpcol or --chi2col."
                    )
                if found:
                    logger.info("Using column %r for the p-values", found[0])
                    if kind == "p":
                        pcol = found[0]
                    elif kind == "mlogp":
                        mlogpcol = found[0]
                    else:
                        chi2col = found[0]
                    break
            else:
                raise ValueError(
                    f"No p-value column: none named {DEFAULT_P!r}, or a p-value, "
                    "-log10(p) or chi-squared spelling (P, LOG10P, CHISQ, …).  "
                    "Name it with --pcol, --mlogpcol or --chi2col.  "
                    f"Available columns: {list(data.columns)}"
                )

    if pcol is not None:
        return neglogp_from_p(data[pcol]), pcol
    if mlogpcol is not None:
        find_column(data, mlogpcol, mlogpcol, (), "--mlogpcol")
        return check_neglogp(data[mlogpcol], mlogpcol), mlogpcol

    find_column(data, chi2col, chi2col, (), "--chi2col")
    dfs = _chi2_df(data, chi2df)
    return neglogp_from_chi2(data[chi2col], dfs), chi2col


def _chi2_df(data: pd.DataFrame, chi2df: str | float) -> np.ndarray | float:
    try:
        value = float(chi2df)
    except (TypeError, ValueError):
        find_column(data, str(chi2df), str(chi2df), (), "--chi2df")
        return np.asarray(data[str(chi2df)], dtype=float)
    if value <= 0:
        raise ValueError(f"--chi2df must be positive, not {chi2df!r}.")
    return value


# ---------------------------------------------------------------------------
# Chromosomes
# ---------------------------------------------------------------------------

_CHR_PREFIX = re.compile(r"^chr", re.IGNORECASE)
_SEX_MITO = {"X": 1, "Y": 2, "XY": 3, "M": 4, "MT": 4}


def chrom_label(value) -> str:
    """Display label: no ``chr`` prefix, ``1.0`` → ``1``, ``x`` → ``X``."""
    s = _CHR_PREFIX.sub("", str(value).strip())
    try:
        f = float(s)
    except ValueError:
        return s.upper() if s.upper() in _SEX_MITO else s
    return str(int(f)) if f.is_integer() else s


def chrom_sort_key(label: str) -> tuple:
    """1, 2, …, 22, then X, Y, XY, MT, then anything else alphabetically."""
    if label.isdigit():
        return (0, int(label), "")
    if label in _SEX_MITO:
        return (1, _SEX_MITO[label], "")
    return (2, 0, label)


def chrom_order(labels) -> list[str]:
    """The distinct chromosome labels in genome order."""
    return sorted(set(labels), key=chrom_sort_key)


# ---------------------------------------------------------------------------
# Genome-wide x positions
# ---------------------------------------------------------------------------


def chrom_index(chrom) -> tuple[np.ndarray, list[str]]:
    """Each row's chromosome as a number: its place in genome order.

    Returns (index per row, the labels in genome order); a missing
    chromosome is -1.  Labels are worked out once per distinct value, so
    ten million rows take about a second.
    """
    codes, uniques = pd.factorize(np.asarray(chrom, dtype=object))
    # several raw values can share a label ("chr1", "1", 1.0)
    unique_labels = [chrom_label(u) for u in uniques]
    order = chrom_order(unique_labels)
    rank = {c: i for i, c in enumerate(order)}
    lab = np.array([rank[u] for u in unique_labels] + [-1], dtype=np.int64)
    return lab[codes], order  # code -1 (missing) → the last entry, -1


def genome_positions(
    chrom: pd.Series | np.ndarray,
    pos: pd.Series | np.ndarray,
    gap: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[float, float]]]:
    """Lay chromosomes end to end on one x axis.

    Each chromosome spans its observed [min, max] position, in genome order,
    separated by *gap* times the total span.  Returns (x per row, tick
    centres, tick labels, (start, end) of each chromosome on the axis).
    """
    label_idx, order = chrom_index(chrom)
    pos = np.asarray(pos, dtype=float)

    ok = (label_idx >= 0) & np.isfinite(pos)
    n = len(order)
    lo = np.full(n, np.inf)
    hi = np.full(n, -np.inf)
    np.minimum.at(lo, label_idx[ok], pos[ok])
    np.maximum.at(hi, label_idx[ok], pos[ok])
    empty = ~np.isfinite(lo)
    lo[empty] = hi[empty] = 0.0
    width = hi - lo
    step = gap * (width.sum() or 1.0)
    start = np.concatenate([[0.0], np.cumsum(width + step)[:-1]])

    x = np.full(len(pos), np.nan)
    x[ok] = pos[ok] - lo[label_idx[ok]] + start[label_idx[ok]]
    centres = start + width / 2
    bounds = [(float(s), float(s + w)) for s, w in zip(start, width, strict=True)]
    return x, centres, order, bounds


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def thin_mask(
    x: np.ndarray,
    y: np.ndarray,
    keep_above: float = 2.0,
    xbins: int = 4000,
    ybins: int = 1000,
    groups: np.ndarray | None = None,
) -> np.ndarray:
    """Rows to draw: every point with y >= *keep_above*, one per grid cell below.

    The grid (*xbins* by *ybins* over the data's range) is finer than a
    screen or print can show, so the plot looks the same; a scan of ten
    million variants, most of them with p > 0.01, draws a few hundred
    thousand points.  With *groups*, each group (colour) is thinned on its
    own so no colour hides another.  Points with NaN x or y are dropped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    keep = ok & (y >= keep_above)
    low = ok & ~keep
    if not low.any():
        return keep

    idx = np.flatnonzero(low)

    def _cell(v, n):
        lo, hi = v[ok].min(), v[ok].max()
        width = (hi - lo) or 1.0
        return np.floor((v[idx] - lo) / width * (n - 1)).astype(np.int64)

    cell = _cell(x, xbins) * ybins + _cell(y, ybins)
    key = pd.DataFrame({"cell": cell})
    if groups is not None:
        key["group"] = np.asarray(groups, dtype=object)[idx]
    first = ~key.duplicated().to_numpy()
    keep[idx[first]] = True
    return keep


# ---------------------------------------------------------------------------
# Example data (dftk dataset gwas)
# ---------------------------------------------------------------------------

# GRCh38 chromosome lengths in Mb, rounded.
_CHR_MB = {
    "1": 248, "2": 242, "3": 198, "4": 190, "5": 181, "6": 171, "7": 159,
    "8": 145, "9": 138, "10": 134, "11": 135, "12": 133, "13": 114, "14": 107,
    "15": 102, "16": 90, "17": 83, "18": 80, "19": 59, "20": 64, "21": 47,
    "22": 51, "X": 156,
}  # fmt: skip

# (chromosome, position of the causal variant, chi-squared at that variant)
_PEAKS = [
    ("2", 60_500_000, 48.0),
    ("6", 32_600_000, 380.0),
    ("9", 22_100_000, 95.0),
    ("12", 111_800_000, 60.0),
    ("16", 53_800_000, 160.0),
    ("19", 44_900_000, 34.0),
]


def simulate_gwas(n_background: int = 40_000, seed: int = 20261112) -> pd.DataFrame:
    """A synthetic association scan for examples: no real people or variants.

    Columns: SNP, Chr, pos, Pval, Chisq (1 df), r2.  Background variants are
    null.  Six peaks each have a causal variant (r2 = 1, the top hit of its
    peak, chi-squared as in _PEAKS) and neighbours whose z-scores follow it
    through linkage disequilibrium, z = r * z_lead + sqrt(1 - r^2) * noise,
    with r2 falling off over ~100 kb, capped so the causal variant stays the
    top hit of its peak.  r2 is the r2 with the nearest peak's
    causal variant, given within 1 Mb of it and empty elsewhere.
    Deterministic for a given *seed*.  numpy only (no scipy).
    """
    rng = np.random.default_rng(seed)
    names = list(_CHR_MB)
    weights = np.array([_CHR_MB[c] for c in names], dtype=float)
    chrom = rng.choice(names, size=n_background, p=weights / weights.sum())
    pos = np.array([rng.integers(1, _CHR_MB[c] * 1_000_000) for c in chrom])
    z = rng.standard_normal(n_background)
    r2 = np.full(n_background, np.nan)

    parts = [(chrom, pos, z, r2)]
    for c, lead, chi2 in _PEAKS:
        z_lead = math.sqrt(chi2)
        # background variants near the lead: weak LD with it
        bg = (chrom == c) & (np.abs(pos - lead) <= 1_000_000) & (pos != lead)
        dist = np.abs(pos[bg] - lead)
        r2[bg] = rng.beta(0.7, 8, size=dist.size) * np.exp(-dist / 3e5)
        z[bg] = _ld_z(z_lead, r2[bg], z[bg])
        # dense variants around the lead, the causal one first
        near = rng.normal(lead, 250_000, size=600).astype(np.int64)
        near = near[(near > 0) & (near != lead)]
        dist = np.abs(near - lead)
        r2_near = np.exp(-dist / 100_000) * rng.beta(4, 1.5, size=near.size)
        z_near = _ld_z(z_lead, r2_near, rng.standard_normal(near.size))
        parts.append(
            (
                np.full(near.size + 1, c),
                np.concatenate([[lead], near]),
                np.concatenate([[z_lead], z_near]),
                np.concatenate([[1.0], np.where(dist <= 1_000_000, r2_near, np.nan)]),
            )
        )

    chrom, pos, z, r2 = (np.concatenate(cols) for cols in zip(*parts, strict=True))
    chisq = z**2
    # p = P(chi2_1 > x) = erfc(sqrt(x / 2))
    pval = np.frompyfunc(lambda v: math.erfc(math.sqrt(v / 2)), 1, 1)(chisq)

    out = pd.DataFrame(
        {
            "Chr": chrom,
            "pos": pos.astype(np.int64),
            "Pval": pval.astype(float),
            "Chisq": chisq,
            "r2": np.round(r2, 3),
        }
    )
    out["_k"] = [chrom_sort_key(c) for c in out["Chr"]]
    out = (
        out.drop_duplicates(["Chr", "pos"])
        .sort_values(["_k", "pos"], kind="stable")
        .drop(columns="_k")
        .reset_index(drop=True)
    )
    out.insert(0, "SNP", [f"sim{i + 1}" for i in range(len(out))])
    return out


def _ld_z(z_lead: float, r2: np.ndarray, noise: np.ndarray) -> np.ndarray:
    # Capped just below the lead so the causal variant stays its peak's top
    # hit (in real data a neighbour can beat it by chance; here r2 is
    # measured from the causal variant, so it should be the one on top).
    r = np.sqrt(r2)
    return np.clip(r * z_lead + np.sqrt(1 - r2) * noise, None, 0.98 * z_lead)
