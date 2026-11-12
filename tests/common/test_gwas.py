"""Tests for dftk.common.gwas (shared helpers of the manhattan / locus plots)."""

import logging

import numpy as np
import pandas as pd
import pytest
import scipy.stats as ss

from dftk.common import gwas
from dftk.common.gwas import (
    MAX_NEGLOGP_FROM_P,
    check_neglogp,
    chrom_label,
    chrom_order,
    find_column,
    genome_positions,
    neglogp_from_chi2,
    neglogp_from_p,
    resolve_neglogp,
    simulate_gwas,
    thin_mask,
)

# ---------------------------------------------------------------------------
# Column lookup
# ---------------------------------------------------------------------------


class TestFindColumn:
    def test_default_name_wins(self):
        df = pd.DataFrame(columns=["Chr", "CHROM"])
        assert find_column(df, None, "Chr", gwas.CHR_ALIASES, "--chrcol") == "Chr"

    def test_alias_ignoring_case(self):
        df = pd.DataFrame(columns=["#CHROM", "BP", "P"])
        assert find_column(df, None, "Chr", gwas.CHR_ALIASES, "--chrcol") == "#CHROM"
        assert find_column(df, None, "pos", gwas.POS_ALIASES, "--poscol") == "BP"

    def test_default_in_other_case_found(self):
        df = pd.DataFrame(columns=["CHR", "POS"])
        assert find_column(df, None, "pos", gwas.POS_ALIASES, "--poscol") == "POS"

    def test_given_column_must_exist(self):
        df = pd.DataFrame(columns=["Chr"])
        with pytest.raises(ValueError, match="'chrom'.*--chrcol"):
            find_column(df, "chrom", "Chr", gwas.CHR_ALIASES, "--chrcol")

    def test_several_aliases_is_an_error(self):
        df = pd.DataFrame(columns=["chrom", "CHROMOSOME"])
        with pytest.raises(ValueError, match="Several columns.*--chrcol"):
            find_column(df, None, "Chr", gwas.CHR_ALIASES, "--chrcol")

    def test_none_found_names_the_option(self):
        df = pd.DataFrame(columns=["a", "b"])
        with pytest.raises(ValueError, match="No column 'pos'.*--poscol"):
            find_column(df, None, "pos", gwas.POS_ALIASES, "--poscol")


# ---------------------------------------------------------------------------
# -log10(p)
# ---------------------------------------------------------------------------


class TestNeglogp:
    def test_from_p(self):
        out = neglogp_from_p([1.0, 0.01, 5e-8, np.nan])
        assert out[:3] == pytest.approx([0.0, 2.0, -np.log10(5e-8)])
        assert np.isnan(out[3])

    def test_p_zero_capped_with_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            out = neglogp_from_p([0.0, 0.5])
        assert out[0] == MAX_NEGLOGP_FROM_P
        assert "--chi2col" in caplog.text

    @pytest.mark.parametrize("bad", [-0.1, 1.5])
    def test_p_out_of_range_is_an_error(self, bad):
        with pytest.raises(ValueError, match="outside"):
            neglogp_from_p([0.5, bad])

    def test_mostly_negative_mlogp_is_log10p(self):
        with pytest.raises(ValueError, match="log10\\(p\\).*eval"):
            check_neglogp(np.array([-1.0, -3.0, -0.5]), "logp")

    def test_mlogp_passes_through(self):
        assert check_neglogp(pd.Series([0.0, 2.0, 8.5]), "m") == pytest.approx(
            [0.0, 2.0, 8.5]
        )

    def test_chi2_agrees_with_p(self):
        chi2 = np.array([0.5, 3.84, 30.0, 200.0])
        expected = -np.log10(ss.chi2.sf(chi2, 1))
        assert neglogp_from_chi2(chi2, 1) == pytest.approx(expected, rel=1e-10)

    def test_chi2_past_underflow_is_finite(self):
        # p = 0 as a double, but chi-squared still gives the true size
        assert ss.chi2.sf(2000.0, 1) == 0.0
        exact = 436.043273716073  # mpmath, 50 digits
        assert neglogp_from_chi2([2000.0], 1)[0] == pytest.approx(exact, rel=1e-12)


class TestResolveNeglogp:
    def test_default_pval(self):
        df = pd.DataFrame({"Pval": [0.1, 0.001]})
        y, col = resolve_neglogp(df)
        assert col == "Pval"
        assert y == pytest.approx([1.0, 3.0])

    def test_falls_back_to_mlogp_then_chi2(self):
        y, col = resolve_neglogp(pd.DataFrame({"LOG10P": [1.0, 3.0]}))
        assert col == "LOG10P"
        assert y == pytest.approx([1.0, 3.0])
        y, col = resolve_neglogp(pd.DataFrame({"CHISQ": [3.84]}))
        assert col == "CHISQ"
        assert y[0] == pytest.approx(-np.log10(0.05), rel=1e-3)

    def test_p_spelling_beats_chi2(self):
        _, col = resolve_neglogp(pd.DataFrame({"chisq": [1.0], "P": [0.5]}))
        assert col == "P"

    def test_explicit_options(self):
        df = pd.DataFrame({"a": [0.01], "b": [2.0], "c": [3.84], "k": [2]})
        assert resolve_neglogp(df, pcol="a")[0] == pytest.approx([2.0])
        assert resolve_neglogp(df, mlogpcol="b")[0] == pytest.approx([2.0])
        y, _ = resolve_neglogp(df, chi2col="c", chi2df="k")
        assert y == pytest.approx(-np.log10(ss.chi2.sf(3.84, 2)))
        y, _ = resolve_neglogp(df, chi2col="c", chi2df="2")
        assert y == pytest.approx(-np.log10(ss.chi2.sf(3.84, 2)))

    def test_only_one_source(self):
        df = pd.DataFrame({"a": [0.01], "b": [2.0]})
        with pytest.raises(ValueError, match="only one"):
            resolve_neglogp(df, pcol="a", mlogpcol="b")

    def test_missing_named_column(self):
        with pytest.raises(ValueError, match="'x'.*--chi2col"):
            resolve_neglogp(pd.DataFrame({"a": [1.0]}), chi2col="x")

    def test_no_p_column(self):
        with pytest.raises(ValueError, match="No p-value column"):
            resolve_neglogp(pd.DataFrame({"a": [1.0]}))

    def test_bad_chi2df(self):
        df = pd.DataFrame({"c": [3.84]})
        with pytest.raises(ValueError, match="positive"):
            resolve_neglogp(df, chi2col="c", chi2df=0)
        with pytest.raises(ValueError, match="'dof'.*--chi2df"):
            resolve_neglogp(df, chi2col="c", chi2df="dof")


# ---------------------------------------------------------------------------
# Chromosomes and positions
# ---------------------------------------------------------------------------


class TestChromosomes:
    @pytest.mark.parametrize(
        "raw,label",
        [("chr1", "1"), ("CHR12", "12"), (3, "3"), (4.0, "4"), ("x", "X"),
         ("chrMT", "MT"), ("Un_gl000220", "Un_gl000220")],
    )  # fmt: skip
    def test_label(self, raw, label):
        assert chrom_label(raw) == label

    def test_order(self):
        labels = ["X", "10", "2", "MT", "1", "Y", "22", "random", "XY"]
        assert chrom_order(labels) == [
            "1", "2", "10", "22", "X", "Y", "XY", "MT", "random",
        ]  # fmt: skip


class TestGenomePositions:
    def test_chromosomes_end_to_end_in_order(self):
        chrom = ["2", "1", "1", "2", "X"]
        pos = [100, 10, 60, 300, 5]
        x, centres, labels, bounds = genome_positions(chrom, pos, gap=0.0)
        assert labels == ["1", "2", "X"]
        # chr1 spans 10..60 → 0..50; chr2 100..300 → 50..250; X a point at 250
        assert x == pytest.approx([50, 0, 50, 250, 250])
        assert bounds == [(0, 50), (50, 250), (250, 250)]
        assert centres == pytest.approx([25, 150, 250])

    def test_gap_and_mixed_spellings(self):
        x, _, labels, _ = genome_positions(["chr1", 1, "1", "chr2"], [0, 50, 100, 0])
        assert labels == ["1", "2"]
        assert x[3] == pytest.approx(100 + 0.01 * 100)

    def test_missing_chromosome_or_position_is_nan(self):
        x, _, labels, _ = genome_positions(["1", None, "1"], [5, 7, np.nan])
        assert labels == ["1"]
        assert x[0] == 0.0
        assert np.isnan(x[1]) and np.isnan(x[2])


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


class TestThinMask:
    def test_keeps_every_point_above_threshold(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1e9, 200_000)
        y = -np.log10(rng.uniform(size=x.size))
        keep = thin_mask(x, y, keep_above=2.0, xbins=500, ybins=200)
        assert keep[y >= 2].all()
        assert keep.sum() < 0.3 * x.size

    def test_duplicates_below_threshold_collapse(self):
        x = np.array([0.0, 0.0, 0.0, 100.0, 50.0])
        y = np.array([0.5, 0.5, 0.5, 0.5, 10.0])
        keep = thin_mask(x, y, keep_above=2.0)
        assert keep.tolist() == [True, False, False, True, True]

    def test_groups_thinned_separately(self):
        x = np.array([0.0, 0.0, 100.0])
        y = np.array([0.5, 0.5, 0.5])
        keep = thin_mask(x, y, groups=np.array(["a", "b", "a"]))
        assert keep.tolist() == [True, True, True]

    def test_nan_dropped(self):
        keep = thin_mask(np.array([np.nan, 1.0, 2.0]), np.array([5.0, np.nan, 1.0]))
        assert keep.tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# Example data
# ---------------------------------------------------------------------------


class TestSimulateGwas:
    @pytest.fixture(scope="class")
    @classmethod
    def data(cls):
        return simulate_gwas()

    def test_columns_and_order(self, data):
        assert list(data.columns) == ["SNP", "Chr", "pos", "Pval", "Chisq", "r2"]
        assert data["SNP"].is_unique
        assert list(pd.unique(data["Chr"])) == chrom_order(data["Chr"])
        assert not data.duplicated(["Chr", "pos"]).any()

    def test_deterministic(self, data):
        pd.testing.assert_frame_equal(data, simulate_gwas())

    def test_pval_matches_chisq(self, data):
        expected = ss.chi2.sf(data["Chisq"], 1)
        assert data["Pval"].to_numpy() == pytest.approx(expected, rel=1e-9)

    def test_background_is_null(self, data):
        far = data["r2"].isna()
        # median of a 1-df chi-squared is 0.455
        assert data.loc[far, "Chisq"].median() == pytest.approx(0.455, abs=0.03)

    def test_causal_variant_tops_each_peak(self, data):
        for chrom, lead, chi2 in gwas._PEAKS:
            sub = data[data["Chr"] == chrom]
            top = sub.loc[sub["Pval"].idxmin()]
            assert top["pos"] == lead
            assert top["r2"] == 1.0
            assert top["Chisq"] == pytest.approx(chi2)


# ---------------------------------------------------------------------------
# QQ plot helpers
# ---------------------------------------------------------------------------


class TestQq:
    def test_expected(self):
        assert gwas.qq_expected(3) == pytest.approx(-np.log10([0.25, 0.5, 0.75]))

    def test_lambda_null_is_one(self):
        p = np.random.default_rng(0).uniform(size=200_000)
        assert gwas.inflation_lambda(-np.log10(p)) == pytest.approx(1.0, abs=0.02)

    def test_lambda_scales_with_chi2(self):
        chi2 = ss.chi2.rvs(1, size=200_000, random_state=1) * 1.2
        lam = gwas.inflation_lambda(neglogp_from_chi2(chi2, 1))
        assert lam == pytest.approx(1.2, abs=0.02)

    def test_lambda_other_df(self):
        chi2 = ss.chi2.rvs(3, size=200_000, random_state=2)
        assert gwas.inflation_lambda(neglogp_from_chi2(chi2, 3), 3) == pytest.approx(
            1.0, abs=0.02
        )

    def test_lambda_ignores_underflow_and_nan(self):
        y = np.array([0.1, 0.3, MAX_NEGLOGP_FROM_P, np.nan, np.inf])
        assert np.isfinite(gwas.inflation_lambda(y))
        assert np.isnan(gwas.inflation_lambda(np.array([np.nan])))

    def test_band_contains_the_expectation(self):
        x, lo, hi = gwas.qq_band(10_000)
        assert np.all(lo <= x) and np.all(x <= hi)
        assert len(x) <= 500
        x95, lo95, hi95 = gwas.qq_band(10_000, 0.95)
        _, lo50, hi50 = gwas.qq_band(10_000, 0.5)
        assert np.all(hi50 <= hi95) and np.all(lo50 >= lo95)

    def test_band_small_n_uses_every_rank(self):
        x, _, _ = gwas.qq_band(5)
        assert x == pytest.approx(gwas.qq_expected(5))


def test_example_data_lambda_near_one():
    # most variants are null, as in a real scan
    data = simulate_gwas()
    assert gwas.inflation_lambda(neglogp_from_p(data["Pval"])) < 1.05
