"""
stattools.common.seq — DNA/sequence utility functions.

Row-wise functions for use with pandas .apply(), importable by any command.

Available functions
-------------------
gc_content(x)       GC percentage of a nucleotide string.
letter_count(x)     Tuple (A, T, G, C) counts.
letter_total(x)     Total ATGC count (non-ATGC characters excluded).
at_gc_ratio(x)      (A+T) / (G+C) ratio.
count_motif(pattern, seq)  Count (possibly overlapping) occurrences of a
                    nucleotide pattern; N in the pattern matches any base.
read_offset(read_pos, ref_pos, cigar, read_seq, count)
                    Extract bases from a short read at a given reference
                    position, accounting for CIGAR-encoded alignment gaps.
"""

import re

import numpy as np

__all__ = [
    "gc_content",
    "letter_count",
    "letter_total",
    "at_gc_ratio",
    "count_motif",
    "read_offset",
]


# ---------------------------------------------------------------------------
# Base composition
# ---------------------------------------------------------------------------


def letter_count(x: str) -> tuple[int, int, int, int]:
    """Return (A, T, G, C) counts for nucleotide string *x*."""
    letters = list(str(x).upper())
    return (
        letters.count("A"),
        letters.count("T"),
        letters.count("G"),
        letters.count("C"),
    )


def gc_content(x: str) -> float:
    """GC percentage of nucleotide string *x*.

    Returns NaN for strings that contain no ATGC characters.
    """
    A, T, G, C = letter_count(x)
    total = A + T + G + C
    return 100.0 * (G + C) / total if total > 0 else np.nan


def letter_total(x: str) -> int:
    """Total ATGC character count in *x*."""
    A, T, G, C = letter_count(x)
    return A + T + G + C


def at_gc_ratio(x: str) -> float:
    """(A+T) / (G+C) ratio of *x*.

    Returns NaN when G+C == 0.
    """
    A, T, G, C = letter_count(x)
    gc = G + C
    return (A + T) / gc if gc > 0 else np.nan


# ---------------------------------------------------------------------------
# Motif counting
# ---------------------------------------------------------------------------


def count_motif(pattern: str, seq: str) -> int:
    """Count occurrences of *pattern* in *seq*.

    'N' in *pattern* matches any single character (A, T, G, or C).
    Overlapping matches are counted.
    """
    regex = pattern.replace("N", "(A|T|C|G)")
    return len(re.findall(regex, str(seq)))


# ---------------------------------------------------------------------------
# CIGAR-based read offset
# ---------------------------------------------------------------------------


def read_offset(
    read_pos: int,
    ref_pos: int,
    cigar: str,
    read_seq: str,
    count: int = 1,
) -> str:
    """Extract *count* bases from a short read at reference position *ref_pos*.

    Handles M (match/mismatch), I (insertion), D/N (deletion/skip), and
    S (soft-clip) CIGAR operations.  Returns '.' for positions outside the
    aligned region and 'N' for positions inside a deletion.

    Parameters
    ----------
    read_pos:  leftmost mapping position of the read on the reference (0-based)
    ref_pos:   reference position to extract (0-based)
    cigar:     CIGAR string (e.g. "5S40M2I10M")
    read_seq:  read sequence string
    count:     number of consecutive reference bases to extract
    """
    cigar_re = re.compile(r"(\d+)([A-Z])")
    offset_list: list[int] = []
    last = -1
    got_soft_clip = False

    for steps_str, op in cigar_re.findall(cigar):
        steps = int(steps_str)
        if op == "M":
            offset_list += [last + i + 1 for i in range(steps)]
            last = offset_list[-1]
        elif op == "I":
            offset_list += [-1] * steps  # insertions consume read but not reference
        elif op == "S" and not got_soft_clip:
            offset_list += [-1] * steps
            last = offset_list[-1]
            got_soft_clip = True
        elif op in ("D", "N"):
            last += steps

    ref_pos_list = [x + read_pos if x >= 0 else x for x in offset_list]

    bases: list[str] = []
    for i in range(count):
        target = ref_pos + i
        try:
            idx = ref_pos_list.index(target)
            base = read_seq[idx]
        except ValueError:
            if ref_pos_list and ref_pos_list[0] < target < ref_pos_list[-1]:
                base = "N"
            else:
                base = "."
        bases.append(base)

    return "".join(bases)
