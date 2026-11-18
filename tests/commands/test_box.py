"""
Tests for dftk box.  Figures are saved with the Agg backend; what was
drawn is read back from the axes.
"""

import argparse
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.box_cmd as box_mod  # noqa: E402
from dftk.commands.box_cmd import BoxCommand, _order  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    BoxCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(tmp_path, df, *argv):
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = box_mod.save_or_show
    with patch.object(
        box_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        BoxCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0].axes[0]


def _medians(ax):
    """y of the median lines of a box plot, left to right."""
    out = []
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        # the median: a horizontal line as wide as the box, whose two ends
        # sit at the same y (whisker caps are narrower)
        if len(y) == 2 and y[0] == y[1] and abs(x[1] - x[0]) > 0.3:
            out.append((float(np.mean(x)), float(y[0])))
    return [y for _, y in sorted(out)]


@pytest.fixture
def levels():
    rng = np.random.default_rng(0)
    dosage = np.repeat([0, 1, 2], [60, 30, 10])
    return pd.DataFrame(
        {
            "dosage": dosage,
            "level": 100 / 2.0**dosage * rng.lognormal(0, 0.1, dosage.size),
            "sex": np.tile(["F", "M"], 50),
        }
    )


class TestOrder:
    def test_sorted_numbers(self):
        assert _order(pd.Series([2, 0, 1, 0]), None) == [0, 1, 2]

    def test_given_first_rest_follow(self):
        assert _order(pd.Series(["b", "a", "c"]), ["c"]) == ["c", "a", "b"]

    def test_given_numbers_as_text(self):
        assert _order(pd.Series([0, 1, 2]), ["2", "0"]) == [2, 0, 1]

    def test_unknown_value(self):
        with pytest.raises(ValueError, match=r"\['9'\]"):
            _order(pd.Series([0, 1]), ["9"])


class TestBox:
    def test_one_box_per_category_with_medians(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level")
        assert [t.get_text() for t in ax.get_xticklabels()] == ["0", "1", "2"]
        expected = levels.groupby("dosage")["level"].median().to_list()
        assert _medians(ax) == pytest.approx(expected)

    def test_order(self, tmp_path, levels):
        ax = _run(
            tmp_path, levels, "-x", "dosage", "-y", "level", "--order", "2", "1", "0"
        )
        assert [t.get_text() for t in ax.get_xticklabels()] == ["2", "1", "0"]
        assert _medians(ax)[0] < _medians(ax)[-1]

    def test_no_x_one_box(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-y", "level", "--count")
        assert len(_medians(ax)) == 1
        assert [t.get_text() for t in ax.get_xticklabels()] == ["n=100"]

    def test_count(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "--count")
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["0\n(n=60)", "1\n(n=30)", "2\n(n=10)"]

    def test_points(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "--points")
        n = sum(len(c.get_offsets()) for c in ax.collections)
        assert n == len(levels)

    def test_groups_side_by_side(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "-g", "sex")
        boxes = [p for p in ax.patches if type(p).__name__ == "PathPatch"]
        assert len(boxes) == 6
        assert len({p.get_facecolor() for p in boxes}) == 2  # one colour per group
        legend = [t.get_text() for t in ax.get_legend().get_texts()]
        assert legend == ["F", "M"]
        assert ax.get_legend().get_title().get_text() == "sex"

    def test_groups_with_points(self, tmp_path, levels):
        ax = _run(
            tmp_path, levels, "-x", "dosage", "-y", "level", "-g", "sex", "--points"
        )
        assert sum(len(c.get_offsets()) for c in ax.collections) == len(levels)

    def test_violin(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "--kind", "violin")
        assert len(ax.collections) >= 3  # one body per category

    def test_logy(self, tmp_path, levels):
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "--logy")
        assert ax.get_yscale() == "log"

    def test_missing_rows_left_out(self, tmp_path, levels):
        levels.loc[:9, "level"] = np.nan
        ax = _run(tmp_path, levels, "-x", "dosage", "-y", "level", "--count")
        assert ax.get_xticklabels()[0].get_text() == "0\n(n=50)"


class TestErrors:
    @pytest.mark.parametrize(
        "argv, match",
        [
            (["-y", "nosuch"], "nosuch"),
            (["-y", "sex"], "isn't numeric"),
            (["-y", "level", "--order", "0"], "give -x"),
            (["-y", "level", "-x", "dosage", "--groupcolorder", "F"], "give -g"),
        ],
    )
    def test_errors(self, tmp_path, levels, argv, match):
        with pytest.raises(ValueError, match=match):
            _run(tmp_path, levels, *argv)

    def test_logy_needs_positive(self, tmp_path, levels):
        levels.loc[0, "level"] = 0
        with pytest.raises(ValueError, match="above 0"):
            _run(tmp_path, levels, "-y", "level", "--logy")


class TestHline:
    @staticmethod
    def _reflines(ax):
        """y of the dashed reference lines."""
        return [
            float(line.get_ydata()[0])
            for line in ax.lines
            if line.get_linestyle() == "--"
        ]

    def test_lines_and_labels(self, tmp_path, levels):
        ax = _run(
            tmp_path, levels, "-x", "dosage", "-y", "level",
            "--hline", "50:half", "80",
        )  # fmt: skip
        assert self._reflines(ax) == [50.0, 80.0]
        labels = [t for t in ax.texts if t.get_text().strip() == "half"]
        assert len(labels) == 1
        # drawn above the boxes, on a background, so it can be read over one
        assert labels[0].get_zorder() > 2 and labels[0].get_bbox_patch() is not None

    def test_on_log_axis(self, tmp_path, levels):
        ax = _run(
            tmp_path, levels, "-x", "dosage", "-y", "level", "--logy",
            "--hline", "30:LoD",
        )  # fmt: skip
        assert ax.get_yscale() == "log"
        assert self._reflines(ax) == [30.0]

    def test_ylim_still_applies(self, tmp_path, levels):
        ax = _run(
            tmp_path, levels, "-x", "dosage", "-y", "level",
            "--hline", "500", "-ym", "0", "200",
        )  # fmt: skip
        assert ax.get_ylim() == (0.0, 200.0)

    def test_no_vline(self):
        with pytest.raises(SystemExit):
            _args("-y", "level", "--vline", "1")
