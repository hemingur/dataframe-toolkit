"""
Tests for dftk manhattan.  Figures are saved with the Agg backend; the
points drawn are checked through the scatter calls.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.manhattan_cmd import ManhattanCommand, gwas_table  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    ManhattanCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def scan():
    return pd.DataFrame(
        {
            "Chr": ["2", "1", "1", "X", "2", "1"],
            "pos": [500, 100, 300, 50, 900, 200],
            "Pval": [1e-10, 0.5, 1e-3, 0.2, 1e-40, np.nan],
            "study": ["a", "a", "b", "b", "a", "b"],
        }
    )


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
    """Run the command to a PNG; return the figure it saved."""
    from unittest.mock import patch

    import dftk.commands.manhattan_cmd as mod

    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = mod.save_or_show
    with patch.object(mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)):
        ManhattanCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0]


class TestGwasTable:
    def test_default_columns_and_missing_rows(self, scan):
        out = gwas_table(scan, _args())
        assert len(out) == 5  # the NaN p-value is left out
        assert out["_y"].tolist() == pytest.approx(
            [10, -np.log10(0.5), 3, -np.log10(0.2), 40]
        )

    def test_named_columns(self, scan):
        df = scan.rename(columns={"Chr": "CHROM", "pos": "BP", "Pval": "p_wald"})
        out = gwas_table(df, _args("--pcol", "p_wald"))
        assert len(out) == 5

    def test_chi2_column(self):
        df = pd.DataFrame({"Chr": ["1"], "pos": [1], "stat": [2000.0], "k": [1]})
        out = gwas_table(df, _args("--chi2col", "stat", "--chi2df", "k"))
        assert out["_y"].iloc[0] == pytest.approx(436.043273716073)

    def test_p_columns_exclusive(self):
        with pytest.raises(SystemExit):
            _args("--pcol", "a", "--chi2col", "b")

    def test_nothing_left_is_an_error(self):
        df = pd.DataFrame({"Chr": ["1"], "pos": [1], "Pval": [np.nan]})
        with pytest.raises(ValueError, match="No rows"):
            gwas_table(df, _args())


class TestThreshold:
    @pytest.mark.parametrize("bad", ["0", "1", "x", "-1e-8"])
    def test_bad_values(self, bad):
        with pytest.raises(SystemExit):
            _args("--sig", bad)

    def test_none(self):
        assert _args("--sig", "none", "--suggestive", "NONE").sig is None


class TestExecute:
    def test_points_in_genome_order(self, tmp_path, scan, scatters):
        _run(tmp_path, scan, "--no-thin")
        x, y, kw = scatters[0]
        # rows in file order; chr1 spans 100..300, chr2 starts after it
        assert x[1] < x[2] < x[0] < x[4] < x[3]
        assert y == pytest.approx([10, -np.log10(0.5), 3, -np.log10(0.2), 40])
        assert kw["rasterized"]
        # alternating colours: chr1 and X dark, chr2 light
        assert kw["c"][1] == kw["c"][3] != kw["c"][0]

    def test_chromosome_ticks(self, tmp_path, scan):
        fig = _run(tmp_path, scan.assign(Chr=["chr2", "chr1", "1", "chrX", "2", "1"]))
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "X"]

    def test_threshold_lines(self, tmp_path, scan):
        fig = _run(tmp_path, scan, "--sig", "1e-6", "--suggestive", "none")
        heights = [ln.get_ydata()[0] for ln in fig.axes[0].get_lines()]
        assert heights == pytest.approx([6.0])

    def test_ylim_clips_to_triangles(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "--ylim", "0", "20")
        assert fig.axes[0].get_ylim() == (0.0, 20.0)
        below, above = scatters
        assert below[1].max() <= 20
        assert above[1].tolist() == [20.0]
        assert above[2]["marker"] == "^"

    def test_default_ylim_shows_the_line(self, tmp_path):
        df = pd.DataFrame({"Chr": ["1", "1"], "pos": [1, 2], "Pval": [0.5, 0.1]})
        fig = _run(tmp_path, df)
        assert fig.axes[0].get_ylim()[1] > -np.log10(5e-8)

    def test_thinning_keeps_hits(self, tmp_path, scatters, caplog):
        rng = np.random.default_rng(0)
        n = 50_000
        df = pd.DataFrame(
            {
                "Chr": np.repeat(["1", "2"], n // 2),
                "pos": np.tile(np.arange(n // 2), 2),
                "Pval": np.concatenate([rng.uniform(size=n - 3), [1e-9, 1e-12, 1e-30]]),
            }
        )
        with caplog.at_level(logging.INFO):
            _run(tmp_path, df)
        x, y, _ = scatters[0]
        assert len(x) < n
        assert (y >= 2).sum() == (df["Pval"] <= 0.01).sum()
        assert "Thinned" in caplog.text

    def test_groups_get_colours_and_bands(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "-g", "study")
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["a", "b"]
        assert {kw["color"] for _, _, kw in scatters} == {"#E69F00", "#56B4E9"}
        assert len(fig.axes[0].patches) == 1  # band behind chr2 (every other)

    def test_subgraph_panels_share_positions(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "--subgraphcol", "study", "--no-thin")
        assert [ax.get_title() for ax in fig.axes] == ["a", "b"]
        (xa, _, _), (xb, _, _) = scatters
        # one coordinate system for both panels: chr1 100..300 → 0..200,
        # a gap of 1 % of the 600 bp spanned, chr2 500..900 → 206..606, X 612
        assert xa.tolist() == pytest.approx(
            [206, 0, 606]
        )  # chr2 500, chr1 100, chr2 900
        assert xb.tolist() == pytest.approx([200, 612])  # chr1 300, X 50
        assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()

    def test_missing_group_column(self, tmp_path, scan):
        with pytest.raises(ValueError, match="-g/--groupcol"):
            _run(tmp_path, scan, "-g", "nosuch")

    def test_example_dataset(self, tmp_path):
        from dftk.common.gwas import simulate_gwas

        fig = _run(tmp_path, simulate_gwas(), "--chi2col", "Chisq")
        assert fig.axes[0].get_ylim()[1] > 80
