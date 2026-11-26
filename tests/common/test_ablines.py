"""
Tests for --abline (common.plot.abline, apply_ablines) in scat and line.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.box_cmd import BoxCommand  # noqa: E402
from dftk.commands.hist_cmd import HistCommand  # noqa: E402
from dftk.commands.line_cmd import LineCommand  # noqa: E402
from dftk.commands.scat_cmd import ScatCommand  # noqa: E402
from dftk.common.plot import abline, apply_ablines  # noqa: E402
from tests.common.test_reflines import _run, table  # noqa: E402, F401


def _parse(command, *argv):
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    return parser.parse_args(["t.tsv", "-x", "x", "-y", "y", *argv])


def _ablines(ax):
    """The dashed lines spanning the panel, as (x, y) arrays."""
    return [
        (ln.get_xdata(), ln.get_ydata())
        for ln in ax.lines
        if ln.get_linestyle() == "--" and len(ln.get_xdata()) > 2
    ]


class TestParse:
    @pytest.mark.parametrize(
        "value, parsed",
        [
            ("0:1", (0.0, 1.0, None)),
            ("-1.5:2:expected", (-1.5, 2.0, "expected")),
            ("0:0.15:15%", (0.0, 0.15, "15%")),
            ("1:2:a:b", (1.0, 2.0, "a:b")),
        ],
    )
    def test_values(self, value, parsed):
        assert abline(value) == parsed

    @pytest.mark.parametrize("value", ["1", "a:1", "1:b", ":1"])
    def test_bad_values(self, value):
        with pytest.raises(argparse.ArgumentTypeError, match="INTERCEPT:SLOPE"):
            abline(value)

    def test_not_given(self):
        assert _parse(ScatCommand()).abline is None

    def test_bare_is_identity(self):
        assert _parse(ScatCommand(), "--abline").abline == [(0.0, 1.0, None)]

    def test_repeated_options_add_up(self):
        args = _parse(
            LineCommand(), "--abline", "--abline=-1:2:neg", "--abline", "0:3", "1:4"
        )
        assert args.abline == [
            (0.0, 1.0, None),
            (-1.0, 2.0, "neg"),
            (0.0, 3.0, None),
            (1.0, 4.0, None),
        ]

    @pytest.mark.parametrize("command", [HistCommand(), BoxCommand()])
    def test_not_on_hist_or_box(self, command):
        parser = argparse.ArgumentParser()
        command.add_arguments(parser)
        assert "--abline" not in parser._option_string_actions


class TestApply:
    def test_spans_the_panel_and_keeps_the_limits(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 2, 3], [10, 20, 30])
        before = ax.get_xlim(), ax.get_ylim()
        apply_ablines(ax, argparse.Namespace(abline=[(100.0, -5.0, "far")]))
        assert (ax.get_xlim(), ax.get_ylim()) == before
        [(x, y)] = _ablines(ax)
        assert (x[0], x[-1]) == pytest.approx(before[0])
        assert y == pytest.approx(100 - 5 * x)
        plt.close(fig)

    def test_log_axes_sample_geometrically(self):
        fig, ax = plt.subplots()
        ax.scatter([1, 10, 100], [1, 10, 100])
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_ablines(ax, argparse.Namespace(abline=[(1.0, 0.1, None)]))
        [(x, y)] = _ablines(ax)
        ratios = x[1:] / x[:-1]
        assert ratios == pytest.approx(np.full_like(ratios, ratios[0]))
        assert y == pytest.approx(1 + 0.1 * x)
        plt.close(fig)

    def test_label_inside_the_panel_and_along_the_line(self):
        fig, ax = plt.subplots()
        ax.scatter([0, 10], [0, 10])
        apply_ablines(ax, argparse.Namespace(abline=[(0.0, 1.0, "y = x")]))
        [text] = ax.texts
        assert text.get_text().strip() == "y = x"
        tx, ty = text.get_position()
        assert ty == pytest.approx(tx)
        assert ax.get_xlim()[0] < tx < ax.get_xlim()[1]
        # rotated to the line's angle on screen (not 45: the axes aren't square)
        (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (10, 10)])
        on_screen = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        assert text.get_rotation() == pytest.approx(on_screen, abs=0.5)
        plt.close(fig)

    def test_no_label_when_the_line_is_out_of_view(self):
        fig, ax = plt.subplots()
        ax.scatter([0, 10], [0, 10])
        apply_ablines(ax, argparse.Namespace(abline=[(1000.0, 0.0, "away")]))
        assert len(ax.texts) == 0
        plt.close(fig)

    def test_nothing_without_the_option(self):
        fig, ax = plt.subplots()
        apply_ablines(ax, argparse.Namespace())
        assert len(ax.lines) == 0
        plt.close(fig)


@pytest.mark.parametrize("command", [ScatCommand(), LineCommand()])
def test_plots_draw_ablines(command, table, tmp_path):  # noqa: F811
    ax = _run(command, table, tmp_path, "-x", "x", "-y", "y", "--abline", "0:0.1:tenth")
    [(x, y)] = _ablines(ax)
    assert y == pytest.approx(0.1 * x)
    assert "tenth" in [t.get_text().strip() for t in ax.texts]


def test_scat_panels_each_get_the_line(table, tmp_path):  # noqa: F811
    import pandas as pd

    df = pd.read_csv(table, sep="\t")
    df["g"] = np.where(df["x"] < 10, "a", "b")
    path = tmp_path / "g.tsv"
    df.to_csv(path, sep="\t", index=False)
    parser = argparse.ArgumentParser()
    ScatCommand().add_arguments(parser)
    out = tmp_path / "p.png"
    captured = []
    orig = plt.Figure.savefig

    def _save(fig, *a, **kw):
        captured.append(fig)
        return orig(fig, *a, **kw)

    plt.Figure.savefig = _save
    try:
        ScatCommand().execute(
            parser.parse_args(
                [str(path), "-x", "x", "-y", "y", "--subgraphcol", "g", "--abline"]
                + ["-f", str(out)]
            )
        )
    finally:
        plt.Figure.savefig = orig
    panels = [ax for ax in captured[0].axes if ax.has_data()]
    assert len(panels) == 2
    assert all(len(_ablines(ax)) == 1 for ax in panels)
