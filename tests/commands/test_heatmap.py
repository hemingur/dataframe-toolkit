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


class TestSubgraph:
    """--subgraphcol: one panel per value, shared rows, columns and scale."""

    @pytest.fixture
    def by_sex(self):
        # Female has no (b, y) cell; male has no (a, x) row value
        return pd.DataFrame(
            {
                "sex": ["F", "F", "F", "M", "M", "M"],
                "row": ["a", "a", "b", "a", "b", "b"],
                "col": ["x", "y", "x", "y", "x", "y"],
                "v": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0],
            }
        )

    def _run(self, tmp_path, df, *argv):
        import dftk.commands.heatmap_cmd as heatmap_mod

        data = tmp_path / "in.tsv"
        df.to_csv(data, sep="\t", index=False)
        png = tmp_path / "out.png"
        saved = []
        orig = heatmap_mod.save_or_show
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                heatmap_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
            )
            HeatmapCommand().execute(_args(str(data), *argv, "-f", str(png)))
        assert png.stat().st_size > 0
        return saved[0]

    @staticmethod
    def _panels(fig):
        return [ax for ax in fig.axes if ax.get_visible() and ax.get_title()]

    @staticmethod
    def _cells(ax):
        """The drawn values as an array, NaN for empty cells."""
        mesh = ax.collections[0]
        a = mesh.get_array()
        return np.ma.filled(a.astype(float), np.nan).reshape(-1)

    def test_one_panel_per_value_aligned(self, tmp_path, by_sex):
        fig = self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex",
        )  # fmt: skip
        f, m = self._panels(fig)
        assert (f.get_title(), m.get_title()) == ("F", "M")
        # rows a, b and columns x, y in every panel; missing cells empty
        np.testing.assert_array_equal(self._cells(f), [1.0, 2.0, 3.0, np.nan])
        np.testing.assert_array_equal(self._cells(m), [np.nan, 4.0, 5.0, 10.0])

    def test_shared_scale_and_one_colourbar(self, tmp_path, by_sex):
        fig = self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex",
        )  # fmt: skip
        f, m = self._panels(fig)
        for ax in (f, m):
            assert ax.collections[0].get_clim() == (1.0, 10.0)
        assert len(fig.axes) == 3  # two panels and one colour bar

    def test_vmin_vmax_apply_to_every_panel(self, tmp_path, by_sex):
        fig = self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex", "--vmin", "0", "--vmax", "20",
        )  # fmt: skip
        assert {ax.collections[0].get_clim() for ax in self._panels(fig)} == {
            (0.0, 20.0)
        }

    def test_tick_labels_on_outer_edge(self, tmp_path, by_sex):
        fig = self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex",
        )  # fmt: skip
        f, m = self._panels(fig)
        assert [t.get_text() for t in f.get_yticklabels()] == ["a", "b"]
        assert all(t.get_text() == "" for t in m.get_yticklabels())
        assert m.get_ylabel() == "" and f.get_ylabel() == "row"
        assert [t.get_text() for t in m.get_xticklabels()] == ["x", "y"]

    def test_order_ncols_format(self, tmp_path, by_sex):
        fig = self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex", "--subgraphorder", "M", "F", "--ncols", "1",
            "--subgraphformat", "sex {sex}",
        )  # fmt: skip
        top, bottom = self._panels(fig)
        assert (top.get_title(), bottom.get_title()) == ("sex M", "sex F")
        # one column: only the bottom panel has x tick labels
        assert all(t.get_text() == "" for t in top.get_xticklabels())
        assert [t.get_text() for t in bottom.get_xticklabels()] == ["x", "y"]

    def test_unused_grid_cells_hidden(self, tmp_path, by_sex):
        df = pd.concat([by_sex, by_sex[by_sex.sex == "F"].assign(sex="X")])
        fig = self._run(
            tmp_path, df, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex", "--ncols", "2",
        )  # fmt: skip
        assert len(self._panels(fig)) == 3
        assert sum(not ax.get_visible() for ax in fig.axes) == 1

    def test_even_row_gaps_with_an_empty_cell(self, tmp_path, by_sex):
        # 5 panels in 2 columns: the 4th is the lowest of its column and has
        # tick labels, above an empty cell; the rows stay evenly spaced
        df = pd.concat([by_sex[by_sex.sex == "F"].assign(sex=s) for s in "ABCDE"])
        fig = self._run(
            tmp_path, df, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex", "--ncols", "2",
        )  # fmt: skip
        fig.canvas.draw()
        panels = self._panels(fig)
        assert [t.get_text() for t in panels[3].get_xticklabels()] == ["x", "y"]
        tops = [panels[i].get_position().y1 for i in (0, 2, 4)]
        bottoms = [panels[i].get_position().y0 for i in (0, 2, 4)]
        gap1, gap2 = bottoms[0] - tops[1], bottoms[1] - tops[2]
        assert gap1 == pytest.approx(gap2, abs=1e-3)

    def test_wide_leaves_out_numeric_subgraph_column(self, tmp_path):
        df = pd.DataFrame(
            {
                "gene": ["g1", "g2", "g1", "g2"],
                "genotype": [0, 0, 2, 2],
                "lunch": [1.0, 2.0, 3.0, 4.0],
                "dinner": [5.0, 6.0, 7.0, np.nan],
            }
        )
        fig = self._run(tmp_path, df, "-y", "gene", "--subgraphcol", "genotype")
        p0, p2 = self._panels(fig)
        assert (p0.get_title(), p2.get_title()) == ("0", "2")
        assert [t.get_text() for t in p2.get_xticklabels()] == ["lunch", "dinner"]
        np.testing.assert_array_equal(self._cells(p2), [3.0, 7.0, 4.0, np.nan])

    def test_wide_select_keeps_subgraph_column(self, tmp_path):
        df = pd.DataFrame(
            {"g": ["a", "a"], "s": ["F", "M"], "u": [1.0, 2.0], "w": [3.0, 4.0]}
        )
        fig = self._run(tmp_path, df, "-y", "g", "--subgraphcol", "s", "--select", "w")
        assert [t.get_text() for t in self._panels(fig)[1].get_xticklabels()] == ["w"]

    def test_default_fmt_from_all_panels(self, tmp_path, by_sex, monkeypatch):
        texts = []
        orig = matplotlib.axes.Axes.text
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "text",
            lambda self, x, y, s, **kw: texts.append(s) or orig(self, x, y, s, **kw),
        )
        by_sex.loc[5, "v"] = 10.5  # one fraction, in the M panel only
        self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex", "--annot",
        )  # fmt: skip
        assert texts[:3] == ["1.00", "2.00", "3.00"]

    def test_same_cell_in_different_panels_is_fine(self, tmp_path, by_sex):
        # (a, y) is in both panels: not a duplicate
        self._run(
            tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
            "--subgraphcol", "sex",
        )  # fmt: skip

    def test_duplicate_within_a_panel_is_an_error(self, tmp_path, by_sex):
        df = pd.concat([by_sex, by_sex.iloc[[0]]])
        with pytest.raises(ValueError, match="several rows for the same"):
            self._run(
                tmp_path, df, "-x", "col", "-y", "row", "-v", "v",
                "--subgraphcol", "sex",
            )  # fmt: skip

    @pytest.mark.parametrize("col", ["row", "col", "v"])
    def test_subgraphcol_cannot_be_an_axis(self, tmp_path, by_sex, col):
        with pytest.raises(ValueError, match="is also -x, -y or -v"):
            self._run(
                tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
                "--subgraphcol", col,
            )  # fmt: skip

    def test_missing_subgraphcol(self, tmp_path, by_sex):
        with pytest.raises(ValueError, match="nosuch"):
            self._run(
                tmp_path, by_sex, "-x", "col", "-y", "row", "-v", "v",
                "--subgraphcol", "nosuch",
            )  # fmt: skip

    def test_lower_in_every_panel(self, tmp_path):
        df = pd.DataFrame(
            {
                "s": ["A"] * 4 + ["B"] * 4,
                "c1": ["p", "p", "q", "q"] * 2,
                "c2": ["p", "q", "p", "q"] * 2,
                "r": [1.0, 0.2, 0.2, 1.0, 1.0, -0.4, -0.4, 1.0],
            }
        )
        fig = self._run(
            tmp_path, df, "-x", "c1", "-y", "c2", "-v", "r",
            "--subgraphcol", "s", "--lower",
        )  # fmt: skip
        for ax in self._panels(fig):
            assert np.isnan(self._cells(ax)[1])  # upper triangle masked
            assert ax.collections[0].get_clim() == (-1.0, 1.0)  # centred on 0
