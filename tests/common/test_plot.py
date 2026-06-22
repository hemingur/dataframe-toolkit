"""
Tests for stattools.common.plot infrastructure.
"""

import argparse

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive for all tests in this file

from stattools.common.plot import (
    WONG_PALETTE,
    apply_style,
    make_figure,
    make_grouplabel,
    parse_figsize,
    subgraph_layout,
)


def _args(**kwargs):
    defaults = dict(fontsize="screen", size="5x3.5", styles=None, usetex=False)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestParseFigsize:
    def test_single_preset(self):
        w, h = parse_figsize("single")
        assert w == pytest.approx(3.5)

    def test_double_preset(self):
        w, h = parse_figsize("double")
        assert w == pytest.approx(7.2)

    def test_full_alias(self):
        assert parse_figsize("full") == parse_figsize("double")

    def test_explicit_wxh(self):
        w, h = parse_figsize("6x4")
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(4.0)

    def test_bad_string_returns_default(self):
        w, h = parse_figsize("nonsense")
        assert w == 5.0
        assert h == 3.5


class TestApplyStyle:
    def test_publication_preset(self):
        import matplotlib as mpl

        apply_style(_args(fontsize="publication"))
        assert mpl.rcParams["xtick.labelsize"] == 6
        assert mpl.rcParams["axes.labelsize"] == 8
        assert mpl.rcParams["axes.titlesize"] == 9

    def test_screen_preset(self):
        import matplotlib as mpl

        apply_style(_args(fontsize="screen"))
        assert mpl.rcParams["xtick.labelsize"] == 9
        assert mpl.rcParams["axes.labelsize"] == 11

    def test_presentation_preset(self):
        import matplotlib as mpl

        apply_style(_args(fontsize="presentation"))
        assert mpl.rcParams["xtick.labelsize"] == 12

    def test_unknown_preset_falls_back_to_screen(self):
        import matplotlib as mpl

        apply_style(_args(fontsize="bogus"))
        assert mpl.rcParams["xtick.labelsize"] == 9


class TestMakeFigure:
    def test_returns_fig_and_2d_axes(self):
        import matplotlib.pyplot as plt

        fig, axes = make_figure(_args())
        assert axes.shape == (1, 1)
        plt.close(fig)

    def test_grid_shape(self):
        import matplotlib.pyplot as plt

        fig, axes = make_figure(_args(), nrows=2, ncols=3)
        assert axes.shape == (2, 3)
        plt.close(fig)


class TestSubgraphLayout:
    def test_single_group(self):
        assert subgraph_layout(1) == (1, 1)

    def test_three_groups_default_cols(self):
        nrows, ncols = subgraph_layout(3)
        assert ncols <= 3
        assert nrows * ncols >= 3

    def test_explicit_ncols(self):
        nrows, ncols = subgraph_layout(6, ncols_hint=2)
        assert ncols == 2
        assert nrows == 3

    def test_more_groups_than_default_cols(self):
        nrows, ncols = subgraph_layout(7)
        assert nrows * ncols >= 7


class TestMakeGrouplabel:
    def test_scalar(self):
        assert make_grouplabel("A") == "A"

    def test_tuple_no_cols(self):
        assert make_grouplabel(("A", "B")) == "A  B"

    def test_tuple_with_cols(self):
        assert make_grouplabel(("A", "1"), groupcols=["group", "rep"]) == "A  1"

    def test_tuple_with_format(self):
        label = make_grouplabel(
            ("A", "1"), groupcols=["group", "rep"], fmt="{group}={rep}"
        )
        assert label == "A=1"

    def test_single_col_tuple_with_format(self):
        label = make_grouplabel(("Jan",), groupcols=["month"], fmt="Month: {month}")
        assert label == "Month: Jan"

    def test_numeric(self):
        assert make_grouplabel(42) == "42"


class TestWongPalette:
    def test_has_eight_colours(self):
        assert len(WONG_PALETTE) == 8

    def test_all_hex(self):
        for c in WONG_PALETTE:
            assert c.startswith("#")
            assert len(c) == 7
