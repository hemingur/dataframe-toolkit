"""
Tests for dftk heatmap.  Figures are saved with the Agg backend.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.heatmap_cmd import HeatmapCommand, _fmt, _matrix  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    HeatmapCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def long_df():
    # The shape corr --all gives: every pair both ways, first-seen order b, a
    return pd.DataFrame(
        {
            "col1": ["b", "b", "a", "a"],
            "col2": ["b", "a", "b", "a"],
            "r": [1.0, -0.5, -0.5, 1.0],
        }
    )


@pytest.fixture
def wide_df():
    return pd.DataFrame(
        {"day": ["Thur", "Fri"], "lunch": [2.77, np.nan], "dinner": [3.0, 2.9]}
    )


class TestMatrix:
    def test_long_keeps_first_seen_order(self, long_df):
        m = _matrix(long_df, _args("-x", "col1", "-y", "col2", "-v", "r"))
        assert list(m.index) == ["b", "a"]
        assert list(m.columns) == ["b", "a"]
        assert m.loc["a", "b"] == -0.5

    def test_wide_uses_other_numeric_columns(self, wide_df):
        m = _matrix(wide_df, _args("-y", "day"))
        assert list(m.index) == ["Thur", "Fri"]
        assert list(m.columns) == ["lunch", "dinner"]
        assert np.isnan(m.loc["Fri", "lunch"])

    def test_duplicate_cells_are_an_error(self, long_df):
        df = pd.concat([long_df, long_df.iloc[[0]]])
        with pytest.raises(ValueError, match="several rows for the same"):
            _matrix(df, _args("-x", "col1", "-y", "col2", "-v", "r"))

    def test_partial_long_arguments_are_an_error(self, long_df):
        with pytest.raises(ValueError, match="needs all of -x, -y and -v"):
            _matrix(long_df, _args("-x", "col1", "-y", "col2"))

    def test_no_arguments_is_an_error(self, long_df):
        with pytest.raises(ValueError, match="give -x, -y and -v"):
            _matrix(long_df, _args())

    def test_wide_repeated_rows_are_an_error(self, wide_df):
        df = pd.concat([wide_df, wide_df])
        with pytest.raises(ValueError, match="repeated values"):
            _matrix(df, _args("-y", "day"))


class TestFmt:
    def test_whole_numbers_get_g(self):
        assert _fmt(np.array([44.0, 123.0, np.nan]), None) == "g"

    def test_fractions_get_two_decimals(self):
        assert _fmt(np.array([0.87, 1.0]), None) == ".2f"

    def test_explicit_fmt_wins(self):
        assert _fmt(np.array([0.5]), ".1%") == ".1%"


class TestExecute:
    def _run(self, tmp_path, df, *argv):
        data = tmp_path / "in.tsv"
        df.to_csv(data, sep="\t", index=False)
        png = tmp_path / "out.png"
        HeatmapCommand().execute(_args(str(data), *argv, "-f", str(png)))
        assert png.stat().st_size > 0
        return png

    def test_correlogram_with_annotations(self, tmp_path, long_df, monkeypatch):
        texts = []
        orig = matplotlib.axes.Axes.text
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "text",
            lambda self, x, y, s, **kw: texts.append(s) or orig(self, x, y, s, **kw),
        )
        self._run(tmp_path, long_df, "-x", "col1", "-y", "col2", "-v", "r", "--annot")
        assert texts == ["1.00", "-0.50", "-0.50", "1.00"]

    def test_lower_masks_upper_triangle(self, tmp_path, long_df, monkeypatch):
        texts = []
        orig = matplotlib.axes.Axes.text
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "text",
            lambda self, x, y, s, **kw: texts.append(s) or orig(self, x, y, s, **kw),
        )
        self._run(
            tmp_path,
            long_df,
            "-x",
            "col1",
            "-y",
            "col2",
            "-v",
            "r",
            "--annot",
            "--lower",
        )
        assert texts == ["1.00", "-0.50", "1.00"]

    def test_lower_needs_square(self, tmp_path, wide_df):
        with pytest.raises(ValueError, match="square"):
            self._run(tmp_path, wide_df, "-y", "day", "--select", "dinner", "--lower")

    def test_both_signs_center_on_zero(self, tmp_path, long_df, monkeypatch):
        meshes = []
        orig = matplotlib.axes.Axes.pcolormesh
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "pcolormesh",
            lambda self, *a, **kw: meshes.append(kw) or orig(self, *a, **kw),
        )
        self._run(tmp_path, long_df, "-x", "col1", "-y", "col2", "-v", "r")
        assert meshes[0]["cmap"] == "RdBu_r"
        assert (meshes[0]["vmin"], meshes[0]["vmax"]) == (-1.0, 1.0)

    def test_wide_select(self, tmp_path, wide_df):
        self._run(tmp_path, wide_df, "-y", "day", "--select", "dinner")
        plt.close("all")
