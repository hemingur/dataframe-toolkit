"""
Tests for stattools.common.seq — DNA/sequence utility functions.
"""

import math

import pytest

from stattools.common.seq import (
    at_gc_ratio,
    count_motif,
    gc_content,
    letter_count,
    letter_total,
    read_offset,
)

# ---------------------------------------------------------------------------
# letter_count
# ---------------------------------------------------------------------------


class TestLetterCount:
    def test_balanced(self):
        A, T, G, C = letter_count("AATTGGCC")
        assert (A, T, G, C) == (2, 2, 2, 2)

    def test_all_one_base(self):
        A, T, G, C = letter_count("AAAA")
        assert A == 4
        assert T == G == C == 0

    def test_case_insensitive(self):
        assert letter_count("atgc") == letter_count("ATGC")

    def test_non_atgc_ignored(self):
        A, T, G, C = letter_count("ATGCNNN")
        assert (A, T, G, C) == (1, 1, 1, 1)


# ---------------------------------------------------------------------------
# gc_content
# ---------------------------------------------------------------------------


class TestGcContent:
    def test_fifty_percent(self):
        assert gc_content("ATGCATGC") == pytest.approx(50.0)

    def test_all_gc(self):
        assert gc_content("GGGGCCCC") == pytest.approx(100.0)

    def test_all_at(self):
        assert gc_content("AAAATTTT") == pytest.approx(0.0)

    def test_empty_string_is_nan(self):
        assert math.isnan(gc_content(""))

    def test_no_atgc_is_nan(self):
        assert math.isnan(gc_content("NNNN"))


# ---------------------------------------------------------------------------
# letter_total
# ---------------------------------------------------------------------------


class TestLetterTotal:
    def test_pure_sequence(self):
        assert letter_total("ATGCATGC") == 8

    def test_with_non_atgc(self):
        assert letter_total("ATGCNNNN") == 4


# ---------------------------------------------------------------------------
# at_gc_ratio
# ---------------------------------------------------------------------------


class TestAtGcRatio:
    def test_equal_ratio(self):
        assert at_gc_ratio("AAGG") == pytest.approx(1.0)

    def test_all_at_returns_nan(self):
        assert math.isnan(at_gc_ratio("AAAA"))

    def test_all_gc(self):
        # AT=0, GC>0 → ratio = 0
        assert at_gc_ratio("GGGG") == pytest.approx(0.0)

    def test_known_ratio(self):
        # "AAATG": A=3, T=1, G=1, C=0 → AT=4, GC=1 → ratio=4.0
        assert at_gc_ratio("AAATG") == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# count_motif
# ---------------------------------------------------------------------------


class TestCountMotif:
    def test_exact_match(self):
        assert count_motif("ATG", "ATGATG") == 2

    def test_n_wildcard(self):
        # re.findall is non-overlapping; "ATGATG" has two non-overlapping NNN matches
        assert count_motif("NNN", "ATGATG") == 2  # ATG, ATG

    def test_no_match(self):
        assert count_motif("CCC", "ATGATG") == 0

    def test_single_match(self):
        assert count_motif("ATG", "ATGCCCC") == 1

    def test_partial_n_pattern(self):
        # ATN matches ATA, ATC, ATG, ATT
        assert count_motif("ATN", "ATAATCATGATTATG") == 5


# ---------------------------------------------------------------------------
# read_offset
# ---------------------------------------------------------------------------


class TestReadOffset:
    def test_simple_match(self):
        # All 5 bases are M (match), read starts at ref position 100
        # Requesting position 102 → index 2 in read → 'G'
        seq = "ATGCA"
        result = read_offset(100, 102, "5M", seq)
        assert result == "G"

    def test_softclip_offset(self):
        # 2S + 3M: first two read bases are soft-clipped
        # read_pos=100, cigar="2S3M", seq="XXATG"
        # Reference positions: [soft, soft, 100, 101, 102]
        seq = "XXATG"
        result = read_offset(100, 100, "2S3M", seq)
        assert result == "A"

    def test_outside_alignment_returns_dot(self):
        seq = "ATGCA"
        result = read_offset(100, 200, "5M", seq)
        assert result == "."

    def test_multi_base_count(self):
        seq = "ATGCA"
        result = read_offset(100, 101, "5M", seq, count=2)
        assert result == "TG"

    def test_deletion_returns_n(self):
        # cigar: 3M 2D 3M
        # read positions: 0,1,2 (M), then deletion skips 2 ref positions, then 3,4,5 (M)
        # ref positions: 100,101,102 (M), 103,104 (deleted), 105,106,107 (M)
        seq = "ATGCAT"
        result = read_offset(100, 103, "3M2D3M", seq)
        assert result == "N"

    # -- insertion tests -------------------------------------------------------

    def test_insertion_middle_m_bases_before(self):
        # 3M 2I 3M, read_pos=100
        # read:  A  C  G  T  T  A  G  C
        # role:  M  M  M  I  I  M  M  M
        # ref:  100 101 102 --  -- 103 104 105
        seq = "ACGTTAGC"
        assert read_offset(100, 100, "3M2I3M", seq) == "A"
        assert read_offset(100, 101, "3M2I3M", seq) == "C"
        assert read_offset(100, 102, "3M2I3M", seq) == "G"

    def test_insertion_middle_m_bases_after(self):
        # Regression: buggy code returned insertion bases ('T') for ref 103/104
        # because it assigned them phantom reference offsets that collided with
        # the subsequent M bases.
        seq = "ACGTTAGC"
        assert read_offset(100, 103, "3M2I3M", seq) == "A"  # was 'T' (buggy)
        assert read_offset(100, 104, "3M2I3M", seq) == "G"  # was 'T' (buggy)
        assert read_offset(100, 105, "3M2I3M", seq) == "C"

    def test_insertion_count_spanning(self):
        # count=2 crossing the insertion: ref 102 → 'G', ref 103 → 'A'
        seq = "ACGTTAGC"
        assert read_offset(100, 102, "3M2I3M", seq, count=2) == "GA"  # was 'GT' (buggy)

    def test_leading_insertion(self):
        # 2I 3M: first two read bases are insertions with no ref position
        # read:  A  B  C  D  E
        # role:  I  I  M  M  M
        # ref:            100 101 102
        seq = "ABCDE"
        assert read_offset(100, 100, "2I3M", seq) == "C"  # was 'A' (buggy)
        assert read_offset(100, 101, "2I3M", seq) == "D"
        assert read_offset(100, 102, "2I3M", seq) == "E"

    def test_insertion_before_queried_pos_is_ignored(self):
        # Position before the insertion is unaffected
        seq = "ACGTTAGC"
        assert read_offset(100, 99, "3M2I3M", seq) == "."  # before read start

    def test_insertion_single_base(self):
        # 4M 1I 4M
        # read:  A  T  G  C  X  T  G  C  A
        # role:  M  M  M  M  I  M  M  M  M
        # ref:  10  11  12  13  --  14  15  16  17
        seq = "ATGCXTGCA"
        assert read_offset(10, 13, "4M1I4M", seq) == "C"
        assert read_offset(10, 14, "4M1I4M", seq) == "T"  # was 'X' (buggy)
        assert read_offset(10, 17, "4M1I4M", seq) == "A"
