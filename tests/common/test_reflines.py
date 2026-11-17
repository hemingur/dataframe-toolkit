"""
Tests for --vline / --hline (common.plot.refline, apply_reflines) and their
use in scat, line, hist and forest.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.forest_cmd import ForestCommand  # noqa: E402
from dftk.commands.hist_cmd import HistCommand  # noqa: E402
from dftk.commands.line_cmd import LineCommand  # noqa: E402
from dftk.commands.scat_cmd import ScatCommand  # noqa: E402
from dftk.common.plot import apply_reflines, refline  # noqa: E402


class TestRefline:
    @pytest.mark.parametrize(
        "value, parsed",
        [
            ("412", (412.0, None)),
            ("-1.5", (-1.5, None)),
            ("2:doubled", (2.0, "doubled")),
            ("1e-3:p = 0.001", (0.001, "p = 0.001")),
            ("5:", (5.0, None)),
        ],
    )
    def test_parse(self, value, parsed):
        assert refline(value) == parsed

    def test_not_a_number(self):
        with pytest.raises(argparse.ArgumentTypeError, match="NUMBER:LABEL"):
            refline("carriers:412")

    def test_draws_lines_and_labels(self):
        fig, ax = plt.subplots()
        ax.plot([0, 10], [0, 10])
        args = argparse.Namespace(
            vline=[(3.0, "three"), (7.0, None)], hline=[(5.0, "five")]
        )
        apply_reflines(ax, args)
        xs = sorted(
            ln.get_xdata()[0]
            for ln in ax.lines[1:]
            if ln.get_xdata()[0] == ln.get_xdata()[1]
        )
        ys = [
            ln.get_ydata()[0]
            for ln in ax.lines[1:]
            if ln.get_ydata()[0] == ln.get_ydata()[1]
            and ln.get_xdata()[0] != ln.get_xdata()[1]
        ]
        assert xs == [3.0, 7.0]
        assert ys == [5.0]
        assert sorted(t.get_text().strip() for t in ax.texts) == ["five", "three"]
        plt.close(fig)

    def test_nothing_without_options(self):
        fig, ax = plt.subplots()
        apply_reflines(ax, argparse.Namespace())
        assert len(ax.lines) == 0
        plt.close(fig)


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.arange(20.0),
            "y": rng.normal(size=20),
            "lab": [f"r{i}" for i in range(20)],
            "se": 0.1,
        }
    )
    path = tmp_path / "t.tsv"
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def _run(command, path, tmp_path, *argv):
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    out = tmp_path / "out.png"
    captured = []
    orig = plt.Figure.savefig

    def _save(fig, *a, **kw):
        captured.append(fig)
        return orig(fig, *a, **kw)

    plt.Figure.savefig = _save
    try:
        command.execute(parser.parse_args([path, *argv, "-f", str(out)]))
    finally:
        plt.Figure.savefig = orig
    assert os.path.getsize(out) > 0
    return captured[0].axes[0]


def _vlines(ax):
    return sorted(
        float(ln.get_xdata()[0])
        for ln in ax.lines
        if len(ln.get_xdata()) == 2
        and ln.get_xdata()[0] == ln.get_xdata()[1]
        and ln.get_linestyle() == "--"
    )


@pytest.mark.parametrize(
    "command, argv",
    [
        (ScatCommand(), ["-x", "x", "-y", "y"]),
        (LineCommand(), ["-x", "x", "-y", "y"]),
        (HistCommand(), ["-x", "y"]),
    ],
)
def test_plots_draw_reflines(command, argv, table, tmp_path):
    ax = _run(command, table, tmp_path, *argv, "--vline", "1.5:a", "--hline", "0.5")
    assert 1.5 in _vlines(ax)
    assert "a" in [t.get_text().strip() for t in ax.texts]


def test_forest_vline_only(table, tmp_path):
    ax = _run(
        ForestCommand(),
        table,
        tmp_path,
        "-y",
        "lab",
        "-x",
        "y",
        "--se",
        "se",
        "--vline",
        "0.25:small",
    )
    assert _vlines(ax) == [0.0, 0.25]  # the reference line and --vline
    parser = argparse.ArgumentParser()
    ForestCommand().add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args([table, "-y", "lab", "-x", "y", "--hline", "1"])


def test_vline_in_view_without_xlim(table, tmp_path):
    ax = _run(ScatCommand(), table, tmp_path, "-x", "x", "-y", "y", "--vline", "50")
    assert ax.get_xlim()[1] >= 50
