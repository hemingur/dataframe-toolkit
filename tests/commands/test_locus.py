"""
Tests for dftk locus.  Figures are saved with the Agg backend; the points
drawn are checked through the scatter calls.
"""

import argparse
import logging
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.locus_cmd as locus_mod  # noqa: E402
from dftk.commands.locus_cmd import LocusCommand, lookup_variant  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    LocusCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def scatters(monkeypatch):
    """Record (x, y, kwargs) of every Axes.scatter call."""
    calls = []
    orig = matplotlib.axes.Axes.scatter

    def _record(self, x, y, *a, **kw):
        calls.append((np.asarray(x, float), np.asarray(y, float), kw))
        return orig(self, x, y, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _record)
    return calls


def _run(tmp_path, df, *argv):
    """Run the command to a PNG; return the axes it drew."""
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = locus_mod.save_or_show
    with patch.object(
        locus_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        LocusCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0].axes[0]


def _texts(ax):
    return [t.get_text() for t in ax.texts]


@pytest.fixture
def region():
    return pd.DataFrame(
        {
            "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
            "Chr": ["6"] * 5,
            "pos": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
            "Pval": [0.5, 1e-6, 1e-12, 1e-3, np.nan],
            "r2": [0.1, 0.5, 1.0, np.nan, 0.9],
        }
    )


class TestLookup:
    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {
                "SNP": ["rs1", "6:200:A:G", "rs3", "rs4"],
                "_chrom": ["chr6", "chr6", "chr6", "chr6"],
                "_pos": [100, 200, 300, 300],
                "_y": [1.0, 2.0, 5.0, 3.0],
            }
        )

    def test_by_id(self, data):
        assert lookup_variant(data, "rs1", "SNP", None) == 0

    def test_by_position_ignores_chr_prefix(self, data):
        assert lookup_variant(data, "6:100", "SNP", None) == 0
        assert lookup_variant(data, "chr6:100", None, "no id") == 0
        assert lookup_variant(data, "CHR6:1_00", None, "no id") == 0

    def test_id_wins_over_position(self, data):
        assert lookup_variant(data, "6:200:A:G", "SNP", None) == 1

    def test_several_at_a_position_smallest_p(self, data, caplog):
        with caplog.at_level(logging.WARNING):
            assert lookup_variant(data, "6:300", "SNP", None) == 2
        assert "2 rows match" in caplog.text

    def test_missing_id(self, data):
        with pytest.raises(ValueError, match="No variant 'rs9' in column 'SNP'"):
            lookup_variant(data, "rs9", "SNP", None)

    def test_missing_position(self, data):
        with pytest.raises(ValueError, match="No variant at 6:999"):
            lookup_variant(data, "6:999", "SNP", None)

    def test_id_without_id_column(self, data):
        with pytest.raises(ValueError, match="no ID column.*why"):
            lookup_variant(data, "rs1", None, "why")


class TestExecute:
    def test_points_in_mb(self, tmp_path, region, scatters):
        _run(tmp_path, region, "--no-thin")
        x, y, _ = scatters[0]
        assert sorted(x) == pytest.approx([1.0, 1.1, 1.2, 1.3])  # NaN p left out
        assert sorted(y) == pytest.approx(sorted(-np.log10([0.5, 1e-6, 1e-12, 1e-3])))

    def test_default_lead_is_smallest_p(self, tmp_path, region, scatters):
        ax = _run(tmp_path, region)
        (lead,) = [c for c in scatters if c[2].get("marker") == "D"]
        assert lead[0] == pytest.approx([1.2]) and lead[1] == pytest.approx([12])
        assert _texts(ax) == ["rs3"]

    def test_lead_by_position_labelled_by_id(self, tmp_path, region, scatters):
        ax = _run(tmp_path, region, "--lead", "chr6:1100000")
        (lead,) = [c for c in scatters if c[2].get("marker") == "D"]
        assert lead[0] == pytest.approx([1.1])
        assert _texts(ax) == ["rs2"]

    def test_labels(self, tmp_path, region, scatters):
        ax = _run(tmp_path, region, "--label", "rs4", "6:1000000", "rs3")
        assert _texts(ax) == ["rs3", "rs4", "rs1"]  # lead first, once
        rings = [c for c in scatters if c[2].get("facecolors") == "none"]
        assert len(rings) == 2

    def test_labelled_points_kept_when_thinning(self, tmp_path, scatters):
        rng = np.random.default_rng(0)
        n = 20_000
        df = pd.DataFrame(
            {
                "SNP": [f"v{i}" for i in range(n)],
                "Chr": "1",
                "pos": np.arange(n) * 10,
                "Pval": rng.uniform(0.02, 1, n),
            }
        )
        df.loc[7, "Pval"] = 1e-9
        _run(tmp_path, df, "--label", "v123")
        x, _, _ = scatters[0]
        assert len(x) < n
        assert 123 * 10 / 1e6 in x

    def test_without_ids_labels_are_positions(self, tmp_path, region):
        ax = _run(tmp_path, region.drop(columns="SNP"), "--label", "6:1000000")
        assert _texts(ax) == ["6:1200000", "6:1000000"]

    def test_id_alias_and_idcol(self, tmp_path, region):
        ax = _run(tmp_path, region.rename(columns={"SNP": "rsid"}), "--lead", "rs2")
        assert _texts(ax) == ["rs2"]
        df = region.rename(columns={"SNP": "name"})
        assert _texts(_run(tmp_path, df, "--idcol", "name")) == ["rs3"]

    def test_ld_colours(self, tmp_path, region, scatters):
        ax = _run(tmp_path, region, "--ldcol", "r2", "--no-thin")
        x, _, kw = scatters[0]
        by_x = dict(zip(np.round(x, 1), kw["c"], strict=True))
        assert by_x[1.0] == locus_mod._LD_COLOURS[0]  # r2 0.1
        assert by_x[1.1] == locus_mod._LD_COLOURS[2]  # 0.5
        assert by_x[1.2] == locus_mod._LD_COLOURS[4]  # 1.0
        assert by_x[1.3] == locus_mod._NO_LD_COLOUR
        assert list(x[-1:]) == pytest.approx([1.2])  # most LD drawn last
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend[0] == "0.8–1" and legend[-1] == "no r²"

    def test_no_ld_no_legend(self, tmp_path, region):
        assert _run(tmp_path, region).get_legend() is None

    def test_xlabel_and_lines(self, tmp_path, region):
        ax = _run(tmp_path, region.assign(Chr="chr6"))
        assert ax.get_xlabel() == "Position on chr6 (Mb)"
        assert [ln.get_ydata()[0] for ln in ax.get_lines()] == pytest.approx(
            [-np.log10(5e-8)]
        )
        ax = _run(tmp_path, region, "--sig", "none", "--suggestive", "1e-5")
        assert [ln.get_ydata()[0] for ln in ax.get_lines()] == pytest.approx([5])

    def test_ylim_caps(self, tmp_path, region, scatters):
        ax = _run(tmp_path, region, "--ylim", "0", "8", "--no-thin")
        assert ax.get_ylim() == (0.0, 8.0)
        capped = [c for c in scatters if c[2].get("marker") == "^"]
        assert capped[0][1].tolist() == [8.0]

    def test_logy(self, tmp_path, region):
        ax = _run(tmp_path, region, "--logy", "5")
        assert ax.get_yscale() == "symlog"

    def test_chi2(self, tmp_path, region, scatters):
        df = region.assign(Chisq=[0.1, 20.0, 50.0, 3.0, 1.0]).drop(columns="Pval")
        _run(tmp_path, df, "--chi2col", "Chisq")
        (lead,) = [c for c in scatters if c[2].get("marker") == "D"]
        assert lead[0] == pytest.approx([1.2])


class TestErrors:
    def test_several_chromosomes(self, tmp_path, region):
        df = pd.concat([region, region.assign(Chr="7", Pval=1e-20)])
        with pytest.raises(ValueError, match="2 chromosomes") as e:
            _run(tmp_path, df)
        assert "Chr == 7 and pos >= 500000 and pos <= 1500000" in str(e.value)

    def test_unknown_lead(self, tmp_path, region):
        with pytest.raises(ValueError, match="rs99"):
            _run(tmp_path, region, "--lead", "rs99")

    def test_missing_idcol(self, tmp_path, region):
        with pytest.raises(ValueError, match="--idcol"):
            _run(tmp_path, region, "--idcol", "nope")

    def test_missing_ldcol(self, tmp_path, region):
        with pytest.raises(ValueError, match="--ldcol"):
            _run(tmp_path, region, "--ldcol", "nope")

    def test_bad_logy(self, tmp_path, region):
        with pytest.raises(ValueError, match="--logy"):
            _run(tmp_path, region, "--logy", "0")

    def test_example_dataset(self, tmp_path):
        from dftk.common.gwas import simulate_gwas

        df = simulate_gwas()
        df = df[(df["Chr"] == "6") & df["pos"].between(31.6e6, 33.6e6)]
        ax = _run(tmp_path, df, "--ldcol", "r2", "--chi2col", "Chisq")
        assert len(_texts(ax)) == 1
