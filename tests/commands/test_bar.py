"""
Tests for dftk bar.  Figures are saved with the Agg backend; what was
drawn is read back from the axes.
"""

import argparse
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.bar_cmd as bar_mod  # noqa: E402
from dftk.commands.bar_cmd import BarCommand, ordered  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    BarCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(tmp_path, df, *argv):
    """Run dftk bar on *df*; return the figure."""
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = bar_mod.save_or_show
    with patch.object(
        bar_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        BarCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0]


def _bars(ax, horizontal=False):
    """(centre, value, colour) of every bar, sorted by position."""
    out = []
    for p in ax.patches:
        x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
        if horizontal:
            out.append((round(y + h / 2, 6), round(w, 6), p.get_facecolor()))
        else:
            out.append((round(x + w / 2, 6), round(h, 6), p.get_facecolor()))
    return sorted(out, key=lambda b: b[0])


@pytest.fixture
def simple():
    return pd.DataFrame(
        {"day": ["Thu", "Fri", "Sat"], "mean": [2.5, 2.0, 3.0], "sem": [0.2, 0.3, 0.1]}
    )


@pytest.fixture
def grouped():
    return pd.DataFrame(
        {
            "gene": ["A", "A", "B", "B", "C"],
            "model": ["LoF", "mis", "LoF", "mis", "mis"],
            "beta": [-0.4, -0.1, 0.3, 0.05, 0.2],
            "se": [0.1, 0.05, 0.1, 0.04, 0.06],
            "lo": [-0.6, -0.2, 0.1, -0.03, 0.08],
            "hi": [-0.2, 0.0, 0.5, 0.13, 0.32],
        }
    )


class TestBars:
    def test_one_bar_per_row_in_file_order(self, tmp_path, simple):
        ax = _run(tmp_path, simple, "-x", "day", "-y", "mean").axes[0]
        assert [(p, v) for p, v, _ in _bars(ax)] == [(0, 2.5), (1, 2.0), (2, 3.0)]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Thu", "Fri", "Sat"]
        assert ax.get_ylim()[0] == 0

    def test_order(self, tmp_path, simple):
        ax = _run(tmp_path, simple, "-x", "day", "-y", "mean", "--order", "Sat").axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Sat", "Thu", "Fri"]
        assert [v for _, v, _ in _bars(ax)] == [3.0, 2.5, 2.0]

    def test_groups_side_by_side_with_gap(self, tmp_path, grouped):
        ax = _run(tmp_path, grouped, "-x", "gene", "-y", "beta", "-g", "model").axes[0]
        bars = _bars(ax)
        # two groups share 0.8 of a slot: centres at ±0.2; C has no LoF bar
        assert [(p, v) for p, v, _ in bars] == [
            (-0.2, -0.4),
            (0.2, -0.1),
            (0.8, 0.3),
            (1.2, 0.05),
            (2.2, 0.2),
        ]
        colours = {p: c for p, _, c in bars}
        assert colours[-0.2] == colours[0.8] != colours[0.2] == colours[2.2]
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["LoF", "mis"]

    def test_groupcolorder(self, tmp_path, grouped):
        ax = _run(
            tmp_path, grouped, "-x", "gene", "-y", "beta", "-g", "model",
            "--groupcolorder", "mis",
        ).axes[0]  # fmt: skip
        assert [p for p, v, _ in _bars(ax) if v == -0.1] == [-0.2]

    def test_horizontal_first_category_on_top(self, tmp_path, simple):
        ax = _run(tmp_path, simple, "-x", "day", "-y", "mean", "--horizontal").axes[0]
        assert [v for _, v, _ in _bars(ax, horizontal=True)] == [2.5, 2.0, 3.0]
        lo, hi = ax.get_ylim()
        assert lo > hi  # inverted: position 0 (Thu) at the top
        assert [t.get_text() for t in ax.get_yticklabels()] == ["Thu", "Fri", "Sat"]


class TestErrorBars:
    def _segments(self, ax):
        (coll,) = [c for c in ax.collections if c.get_segments()]
        return sorted(
            (round(s[0][0], 6), round(s[0][1], 6), round(s[1][1], 6))
            for s in coll.get_segments()
        )

    def test_yerr(self, tmp_path, simple):
        ax = _run(tmp_path, simple, "-x", "day", "-y", "mean", "--yerr", "sem").axes[0]
        assert self._segments(ax) == [(0, 2.3, 2.7), (1, 1.7, 2.3), (2, 2.9, 3.1)]

    def test_yci(self, tmp_path, grouped):
        ax = _run(
            tmp_path, grouped, "-x", "gene", "-y", "beta", "-g", "model",
            "--yci", "lo,hi",
        ).axes[0]  # fmt: skip
        segs = self._segments(ax)
        assert (-0.2, -0.6, -0.2) in segs and (2.2, 0.08, 0.32) in segs

    def test_yci_must_contain_the_value(self, tmp_path, grouped):
        grouped.loc[0, "hi"] = -0.5
        with pytest.raises(ValueError, match="must contain the value"):
            _run(tmp_path, grouped, "-x", "gene", "-y", "beta", "-g", "model",
                 "--yci", "lo,hi")  # fmt: skip

    def test_not_both(self):
        with pytest.raises(SystemExit):
            _args("f.tsv", "-x", "a", "-y", "b", "--yerr", "s", "--yci", "l,h")


class TestLabels:
    def test_past_the_error_bar(self, tmp_path, simple):
        ax = _run(
            tmp_path, simple, "-x", "day", "-y", "mean", "--yerr", "sem",
            "--labels", "--fmt", ".1f",
        ).axes[0]  # fmt: skip
        texts = {t.get_text(): t.xy for t in ax.texts}
        assert texts == {"2.5": (0, 2.7), "2.0": (1, 2.3), "3.0": (2, 3.1)}
        assert ax.get_ylim()[1] > 3.1  # room above the highest label

    def test_negative_below(self, tmp_path, grouped):
        ax = _run(
            tmp_path, grouped, "-x", "gene", "-y", "beta", "-g", "model",
            "--yerr", "se", "--labels", "--fmt", ".2f",
        ).axes[0]  # fmt: skip
        neg = next(t for t in ax.texts if t.get_text() == "-0.40")
        assert neg.xy == pytest.approx((-0.2, -0.5))
        assert neg.get_va() == "top"


class TestPanels:
    @pytest.fixture
    def panels(self, grouped):
        a = grouped.assign(island="X")
        b = grouped.assign(island="Y", beta=grouped["beta"] * 3)
        return pd.concat([a, b.iloc[:3]])

    def test_shared_categories_and_value_axis(self, tmp_path, panels):
        fig = _run(
            tmp_path, panels, "-x", "gene", "-y", "beta", "-g", "model",
            "--subgraphcol", "island",
        )  # fmt: skip
        left, right = [ax for ax in fig.axes if ax.get_visible()]
        assert [left.get_title(), right.get_title()] == ["X", "Y"]
        assert left.get_ylim() == right.get_ylim()
        assert left.get_ylim()[0] <= -1.2  # -0.4 * 3 in panel Y
        assert [t.get_text() for t in right.get_xticklabels()] == ["A", "B", "C"]
        assert right.get_ylabel() == "" and left.get_ylabel() == "beta"

    def test_horizontal_names_on_left_column_only(self, tmp_path, panels):
        fig = _run(
            tmp_path, panels, "-x", "gene", "-y", "beta", "--subgraphcol",
            "island", "-g", "model", "--horizontal",
        )  # fmt: skip
        left, right = [ax for ax in fig.axes if ax.get_visible()]
        assert left.yaxis.get_tick_params()["labelleft"]
        assert not right.yaxis.get_tick_params()["labelleft"]


class TestErrors:
    def test_duplicate_bars(self, tmp_path, simple):
        df = pd.concat([simple, simple.iloc[:1]])
        with pytest.raises(ValueError, match=r"2 rows share a bar \(day\).*stat -g"):
            _run(tmp_path, df, "-x", "day", "-y", "mean")

    def test_not_numeric(self, tmp_path, simple):
        simple["mean"] = ["a", "2", "3"]
        with pytest.raises(ValueError, match="-y column 'mean' isn't numeric"):
            _run(tmp_path, simple, "-x", "day", "-y", "mean")

    def test_rows_without_value_left_out(self, tmp_path, simple):
        simple.loc[1, "mean"] = np.nan
        ax = _run(tmp_path, simple, "-x", "day", "-y", "mean").axes[0]
        assert [v for _, v, _ in _bars(ax)] == [2.5, 3.0]

    @pytest.mark.parametrize(
        "argv, message",
        [
            (["--hline", "1"], "--hline"),
            (["--horizontal", "--hline", "1"], "use --vline"),
            (["--vline", "1"], "use --hline"),
            (["--order", "Mon"], "--order names values not in the data"),
            (["--groupcolorder", "x"], "give -g"),
            (["-g", "day"], "different columns"),
        ],
    )
    def test_bad_options(self, tmp_path, simple, argv, message):
        if argv == ["--hline", "1"]:  # fine: values on the y axis
            _run(tmp_path, simple, "-x", "day", "-y", "mean", *argv)
            return
        with pytest.raises(ValueError, match=message):
            _run(tmp_path, simple, "-x", "day", "-y", "mean", *argv)


def test_ordered_first_appearance():
    assert ordered(pd.Series(["b", "a", "b", None, "c"]), None, "--order") == [
        "b",
        "a",
        "c",
    ]
    assert ordered(pd.Series([2, 1, 3]), ["3"], "--order") == [3, 2, 1]
