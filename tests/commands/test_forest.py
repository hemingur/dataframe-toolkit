"""
Tests for dftk forest.  Figures are saved with the Agg backend; what was
drawn is read back from the axes (tick labels, line collections, texts).
"""

import argparse
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.forest_cmd as forest_mod  # noqa: E402
from dftk.commands.forest_cmd import (  # noqa: E402
    ForestCommand,
    cell_text,
    ci_text,
)


def _args(*argv):
    parser = argparse.ArgumentParser()
    ForestCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(tmp_path, df, *argv):
    """Run the command to a PNG; return the figure."""
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = forest_mod.save_or_show
    with patch.object(
        forest_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        ForestCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0]


def _ylabels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def _intervals(ax):
    """{y: (lo, hi)} of the interval lines."""
    out = {}
    for coll in ax.collections:
        for seg in getattr(coll, "get_segments", lambda: [])():
            (x0, y0), (x1, _) = seg
            out[round(float(y0), 3)] = (float(x0), float(x1))
    return out


def _points(ax):
    """[(x, y)] of the estimate markers (the largest scatter collections)."""
    pts = []
    for coll in ax.collections:
        if hasattr(coll, "get_offsets") and not hasattr(coll, "get_segments"):
            pts.extend(map(tuple, coll.get_offsets()))
    return pts


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "pheno": ["asthma", "gout", "migraine"],
            "category": ["resp", "metab", "neuro"],
            "beta": [0.5, -0.2, 0.0],
            "se": [0.1, 0.05, 0.2],
            "lo": [0.3, -0.3, -0.4],
            "hi": [0.7, -0.1, 0.4],
            "Pval": [1e-6, 0.0001234, 1.0],
            "n": [120, 45, 300],
        }
    )


class TestText:
    def test_ci_text(self):
        assert ci_text(1.234, 1.0, 1.5, 2) == "1.23 [1.00, 1.50]"
        assert ci_text(1.234, np.nan, np.nan, 1) == "1.2"
        assert ci_text(np.nan, 1.0, 2.0, 2) == ""

    @pytest.mark.parametrize(
        "value, text",
        [
            (120, "120"),
            (120.0, "120"),
            (0.0001234, "0.000123"),
            (1.04e-211, "1.04e-211"),
            ("resp", "resp"),
            (np.nan, ""),
            (None, ""),
            (True, "True"),
        ],
    )
    def test_cell_text(self, value, text):
        assert cell_text(value) == text


class TestLayout:
    def test_rows_top_to_bottom_in_file_order(self, tmp_path, results):
        ax = _run(tmp_path, results, "-y", "pheno", "-x", "beta").axes[0]
        assert _ylabels(ax) == ["asthma", "gout", "migraine"]
        lo, hi = ax.get_ylim()
        assert lo > hi  # inverted: row 0 at the top

    def test_se_interval(self, tmp_path, results):
        ax = _run(tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se").axes[0]
        iv = _intervals(ax)
        assert iv[0.0] == pytest.approx((0.5 - 1.959964 * 0.1, 0.5 + 1.959964 * 0.1))

    def test_level(self, tmp_path, results):
        ax = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se",
            "--level", "0.9",
        ).axes[0]  # fmt: skip
        assert _intervals(ax)[0.0][1] == pytest.approx(0.5 + 1.644854 * 0.1)

    def test_ci_columns(self, tmp_path, results):
        ax = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--ci", "lo, hi"
        ).axes[0]
        assert _intervals(ax)[1.0] == pytest.approx((-0.3, -0.1))

    def test_exp(self, tmp_path, results):
        ax = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--ci", "lo,hi", "--exp"
        ).axes[0]
        assert ax.get_xscale() == "log"
        assert _intervals(ax)[0.0] == pytest.approx((np.exp(0.3), np.exp(0.7)))
        assert (np.exp(0.5), 0.0) in [
            (pytest.approx(x), pytest.approx(y)) for x, y in _points(ax)
        ]
        ref = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert ref[0].get_xdata()[0] == 1.0
        assert ax.get_xlabel() == "exp(beta)"

    @pytest.mark.parametrize(
        "argv, lines", [([], [0.0]), (["--ref", "0.2"], [0.2]), (["--ref", "none"], [])]
    )
    def test_ref(self, tmp_path, results, argv, lines):
        ax = _run(tmp_path, results, "-y", "pheno", "-x", "beta", *argv).axes[0]
        assert [line.get_xdata()[0] for line in ax.lines] == lines

    def test_missing_estimate_keeps_label(self, tmp_path, results):
        results.loc[1, "beta"] = np.nan
        ax = _run(tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se").axes[0]
        assert _ylabels(ax) == ["asthma", "gout", "migraine"]
        assert sorted(_intervals(ax)) == [0.0, 2.0]

    def test_sections(self, tmp_path):
        df = pd.DataFrame(
            {
                "sec": ["A", "B", "A"],
                "lab": ["a1", "b1", "a2"],
                "est": [1.0, 2.0, 3.0],
            }
        )
        ax = _run(tmp_path, df, "-y", "lab", "-x", "est", "--section", "sec").axes[0]
        assert _ylabels(ax) == ["A", "a1", "a2", "B", "b1"]
        bold = [t.get_fontweight() == "bold" for t in ax.get_yticklabels()]
        assert bold == [True, False, False, True, False]
        assert sorted(y for _, y in _points(ax)) == [1.0, 2.0, 4.0]

    def test_groups_share_a_row(self, tmp_path):
        df = pd.DataFrame(
            {
                "lab": ["x", "x", "y", "y"],
                "g": ["het", "hom", "het", "hom"],
                "est": [1.0, 2.0, 3.0, 4.0],
            }
        )
        fig = _run(
            tmp_path, df, "-y", "lab", "-x", "est", "-g", "g",
            "--groupcolorder", "hom", "het",
        )  # fmt: skip
        ax = fig.axes[0]
        assert _ylabels(ax) == ["x", "y"]
        ys = dict(_points(ax))
        assert ys[2.0] < ys[1.0] < 0.5  # hom above het, both on row x
        assert 0.5 < ys[4.0] < ys[3.0]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["hom", "het"]

    def test_groups_duplicate_label_is_error(self, tmp_path):
        df = pd.DataFrame({"lab": ["x", "x"], "g": ["a", "a"], "est": [1.0, 2.0]})
        with pytest.raises(ValueError, match="'x' appears more than once"):
            _run(tmp_path, df, "-y", "lab", "-x", "est", "-g", "g")

    def test_xlim_clips_with_arrows(self, tmp_path, results):
        ax = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--ci", "lo,hi",
            "-xm", "-0.25", "0.6",
        ).axes[0]  # fmt: skip
        iv = _intervals(ax)
        assert iv[0.0] == pytest.approx((0.3, 0.6))
        assert iv[1.0] == pytest.approx((-0.25, -0.1))
        markers = [
            (tuple(p), c.get_paths()[0].vertices.round(2).tolist())
            for c in ax.collections
            if hasattr(c, "get_offsets") and not hasattr(c, "get_segments")
            for p in c.get_offsets()
        ]
        edges = {xy for xy, _ in markers}
        assert (0.6, 0.0) in edges and (-0.25, 1.0) in edges


class TestTextColumns:
    def _texts(self, fig):
        return [t.get_text() for t in fig.axes[0].texts]

    def test_text_and_textcol(self, tmp_path, results):
        fig = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--ci", "lo,hi",
            "--text", "--textcol", "Pval", "n",
        )  # fmt: skip
        texts = self._texts(fig)
        assert "0.50 [0.30, 0.70]" in texts
        assert "beta [lo, hi]" in texts
        assert {"Pval", "n", "1e-06", "0.000123", "120", "300"} <= set(texts)
        # the figure grew to the right instead of squeezing the plot
        assert fig.get_size_inches()[0] > 7

    def test_text_headers(self, tmp_path, results):
        fig = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se",
            "--exp", "--text", "--digits", "1",
        )  # fmt: skip
        texts = self._texts(fig)
        assert "exp(beta) [95% CI]" in texts
        assert f"{np.exp(0.5):.1f}" in [t.split(" ")[0] for t in texts]

    def test_group_text_in_series_colour(self, tmp_path):
        df = pd.DataFrame(
            {"lab": ["x", "x"], "g": ["a", "b"], "est": [1.0, 2.0], "se": [0.1, 0.1]}
        )
        fig = _run(
            tmp_path, df, "-y", "lab", "-x", "est", "--se", "se", "-g", "g", "--text"
        )
        colours = {
            t.get_text().split(" ")[0]: t.get_color()
            for t in fig.axes[0].texts
            if "[" in t.get_text() and "CI" not in t.get_text()
        }
        assert colours["1.00"] != colours["2.00"]


class TestWeighted:
    def test_marker_area_by_precision(self, tmp_path, results):
        ax = _run(
            tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se", "--weighted"
        ).axes[0]
        coll = next(c for c in ax.collections if len(c.get_offsets()) == 3)
        sizes = coll.get_sizes()
        # se 0.05 is the most precise: largest; 0.1 has a quarter the weight
        assert sizes[1] == sizes.max()
        assert sizes[0] == pytest.approx(sizes[1] / 4)
        assert sizes[2] == pytest.approx(sizes[1] * 0.1)  # clipped at 10 %

    def test_needs_an_interval(self, tmp_path, results):
        with pytest.raises(ValueError, match="--weighted needs"):
            _run(tmp_path, results, "-y", "pheno", "-x", "beta", "--weighted")


class TestErrors:
    @pytest.mark.parametrize(
        "argv, match",
        [
            (["--ci", "lo"], "two column names"),
            (["--ci", "lo,nosuch"], "nosuch"),
            (["--se", "pheno"], "isn't numeric"),
            (["--level", "95"], "between 0 and 1"),
            (["--ref", "zero"], "number or none"),
            (["--logx"], "above 0"),
            (["--exp", "--logx"], "leave out --logx"),
            (["-xm", "1", "0"], "LO must be below HI"),
            (["--textcol", "nosuch"], "nosuch"),
        ],
    )
    def test_errors(self, tmp_path, results, argv, match):
        with pytest.raises(ValueError, match=match):
            _run(tmp_path, results, "-y", "pheno", "-x", "beta", *argv)

    def test_ci_and_se_exclusive(self):
        with pytest.raises(SystemExit):
            _args("-y", "a", "-x", "b", "--ci", "l,h", "--se", "s")

    def test_negative_se(self, tmp_path, results):
        results.loc[0, "se"] = -1
        with pytest.raises(ValueError, match="negative"):
            _run(tmp_path, results, "-y", "pheno", "-x", "beta", "--se", "se")
