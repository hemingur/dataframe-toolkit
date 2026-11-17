"""
Tests for dftk phewas.  Figures are saved with the Agg backend; what was
drawn is read back from the axes.
"""

import argparse
import logging
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.phewas_cmd as phewas_mod  # noqa: E402
from dftk.commands.phewas_cmd import PhewasCommand, layout  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    PhewasCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(tmp_path, df, *argv):
    """Run the command to a PNG; return the axes it drew."""
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = phewas_mod.save_or_show
    with patch.object(
        phewas_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        PhewasCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0].axes[0]


def _points(ax):
    """[(marker path vertex count, x, y)] of every point drawn."""
    out = []
    for c in ax.collections:
        for x, y in c.get_offsets():
            out.append((len(c.get_paths()[0].vertices), float(x), float(y)))
    return out


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "category": ["Heart", "Heart", "Blood", "Heart", "Eye"],
            "phenotype": ["AF", "CAD", "Anaemia", "Stroke", "Cataract"],
            "Pval": [1e-8, 1e-6, 0.5, 0.2, 1e-20],
            "beta": [0.3, -0.2, 0.0, 0.1, 0.4],
        }
    )


class TestLayout:
    def test_categories_in_order_of_first_appearance(self, results):
        data = results.assign(_cat=results["category"])
        x, spans = layout(data)
        assert [c for c, _, _ in spans] == ["Heart", "Blood", "Eye"]
        # Heart rows keep their file order, then a gap before Blood
        assert list(x) == [0, 1, 4, 2, 6]
        assert spans[0][1:] == (0, 2)

    def test_tick_labels_and_y(self, tmp_path, results):
        ax = _run(tmp_path, results)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Heart", "Blood", "Eye"]
        ys = sorted(y for _, _, y in _points(ax))
        assert ys == pytest.approx(sorted(-np.log10(results["Pval"])))

    def test_category_colours_differ(self, tmp_path, results):
        ax = _run(tmp_path, results)
        colours = {t.get_text(): t.get_color() for t in ax.get_xticklabels()}
        assert len(set(colours.values())) == 3


class TestDirection:
    def test_triangles_by_sign(self, tmp_path, results):
        from matplotlib.markers import MarkerStyle

        def shape(collection):
            verts = collection.get_paths()[0].vertices
            for m in ("^", "v", "o"):
                ms = MarkerStyle(m)
                ref = ms.get_path().transformed(ms.get_transform()).vertices
                if ref.shape == verts.shape and np.allclose(ref, verts):
                    return m
            return None

        ax = _run(tmp_path, results, "--betacol", "beta")
        marker_at = {
            round(float(y), 3): shape(c)
            for c in ax.collections
            for _, y in c.get_offsets()
        }
        assert marker_at[8.0] == "^"  # AF, beta > 0
        assert marker_at[6.0] == "v"  # CAD, beta < 0
        assert marker_at[round(-np.log10(0.5), 3)] == "o"  # Anaemia, beta 0
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["beta > 0", "beta < 0"]

    def test_no_betacol_no_legend(self, tmp_path, results):
        assert _run(tmp_path, results).get_legend() is None

    def test_non_numeric_beta(self, tmp_path, results):
        with pytest.raises(ValueError, match="isn't numeric"):
            _run(tmp_path, results, "--betacol", "phenotype")


class TestLines:
    def test_bonferroni_default(self, tmp_path, results):
        ax = _run(tmp_path, results)
        (line,) = ax.lines
        assert line.get_ydata()[0] == pytest.approx(-np.log10(0.05 / 5))

    def test_fixed_and_suggestive(self, tmp_path, results):
        ax = _run(tmp_path, results, "--sig", "1e-5", "--suggestive", "1e-3")
        assert sorted(ln.get_ydata()[0] for ln in ax.lines) == pytest.approx([3, 5])

    def test_none(self, tmp_path, results):
        ax = _run(tmp_path, results, "--sig", "none")
        assert len(ax.lines) == 0
        assert len(ax.texts) == 0  # no line, so nothing counts as a hit

    def test_bad_sig(self):
        with pytest.raises(SystemExit):
            _args("--sig", "2")


class TestLabels:
    def test_hits_above_line_strongest_first(self, tmp_path, results):
        ax = _run(tmp_path, results)
        assert [t.get_text() for t in ax.texts] == ["Cataract", "AF", "CAD"]

    def test_label_limit(self, tmp_path, results, caplog):
        with caplog.at_level(logging.INFO):
            ax = _run(tmp_path, results, "--label", "1")
        assert [t.get_text() for t in ax.texts] == ["Cataract"]
        assert "the 1 strongest of 3" in caplog.text
        assert len(_run(tmp_path, results, "--label", "0").texts) == 0

    def test_labels_do_not_overlap(self, tmp_path):
        df = pd.DataFrame(
            {
                "category": ["A"] * 4,
                "phenotype": ["first long name", "second long name", "x", "y"],
                "Pval": [1e-9, 2e-9, 3e-9, 0.5],
            }
        )
        ax = _run(tmp_path, df)
        renderer = ax.figure.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        assert len(boxes) == 3
        for i in range(len(boxes)):
            for j in range(i):
                assert not boxes[i].overlaps(boxes[j])


class TestColumns:
    def test_named_columns_and_mlogp(self, tmp_path, results):
        df = results.rename(columns={"category": "grp", "phenotype": "name"})
        df["mlogp"] = -np.log10(df.pop("Pval"))
        ax = _run(tmp_path, df, "-c", "grp", "-l", "name", "--mlogpcol", "mlogp")
        assert {t.get_text() for t in ax.texts} == {"Cataract", "AF", "CAD"}

    def test_p_value_alias(self, tmp_path, results):
        ax = _run(tmp_path, results.rename(columns={"Pval": "P"}))
        assert len(_points(ax)) == 5

    def test_missing_category_column(self, tmp_path, results):
        with pytest.raises(ValueError, match="nosuch"):
            _run(tmp_path, results, "-c", "nosuch")

    def test_rows_without_p_left_out(self, tmp_path, results):
        results.loc[2, "Pval"] = np.nan
        assert len(_points(_run(tmp_path, results))) == 4

    def test_logy(self, tmp_path, results):
        ax = _run(tmp_path, results, "--logy", "5")
        assert ax.get_yscale() == "symlog"

    def test_ylim_caps(self, tmp_path, results):
        ax = _run(tmp_path, results, "-ym", "0", "10")
        assert max(y for _, _, y in _points(ax)) == pytest.approx(10)
