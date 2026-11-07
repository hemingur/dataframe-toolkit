"""
Tests for dftk pairs.  Figures are saved with the Agg backend.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.pairs_cmd import PairsCommand, _columns, _groups  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    PairsCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    return pd.DataFrame(
        {
            "a": a,
            "b": a + rng.normal(size=60),
            "c": -a + rng.normal(size=60),
            "ok": [True, False] * 30,
            "grp": pd.Categorical(["y", "x"] * 30, categories=["x", "y"]),
        }
    )


class TestColumns:
    def test_default_is_every_numeric_column(self, df):
        assert _columns(df, None, exclude=set()) == ["a", "b", "c"]

    def test_names_and_globs_in_order(self, df):
        assert _columns(df, ["c", "[ab]"], exclude=set()) == ["c", "a", "b"]

    def test_group_column_is_excluded(self, df):
        assert _columns(df, None, exclude={"a"}) == ["b", "c"]

    def test_needs_two_columns(self, df):
        with pytest.raises(ValueError, match="at least two"):
            _columns(df, ["a"], exclude=set())

    def test_column_name_with_glob_characters_is_literal(self, df):
        df = df.rename(columns={"a": "a[1]"})
        assert _columns(df, ["a[1]", "b"], exclude=set()) == ["a[1]", "b"]

    def test_glob_matching_nothing(self, df):
        with pytest.raises(ValueError, match="matches no numeric column"):
            _columns(df, ["z*", "a"], exclude=set())


class TestGroups:
    def test_categories_keep_their_order(self, df):
        assert [label for label, _ in _groups(df, "grp", None)] == ["x", "y"]

    def test_groupcolorder(self, df):
        assert [label for label, _ in _groups(df, "grp", ["y"])] == ["y", "x"]

    def test_groupcolorder_skips_missing_group(self, df):
        df.loc[0, "grp"] = np.nan
        assert [label for label, _ in _groups(df, "grp", ["y"])] == ["y", "x"]

    def test_no_group(self, df):
        assert _groups(df, None, None)[0][0] is None


class TestExecute:
    def _tsv(self, tmp_path, df):
        data = tmp_path / "in.tsv"
        df.to_csv(data, sep="\t", index=False)
        return str(data)

    def _run(self, tmp_path, df, *argv):
        png = tmp_path / "out.png"
        PairsCommand().execute(_args(self._tsv(tmp_path, df), *argv, "-f", str(png)))
        assert png.stat().st_size > 0

    @pytest.mark.parametrize(
        "argv",
        [
            (),
            ("-g", "grp"),
            ("-g", "grp", "--diag", "hist", "--upper", "corr"),
            ("-c", "a", "b", "--upper", "corr", "--method", "spearman"),
            ("--size", "4x4", "-t", "Pairs"),
        ],
    )
    def test_runs(self, tmp_path, df, argv):
        self._run(tmp_path, df, *argv)

    def _texts(self, monkeypatch):
        texts = []
        orig = matplotlib.axes.Axes.text
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "text",
            lambda self, x, y, s, **kw: texts.append(s) or orig(self, x, y, s, **kw),
        )
        return texts

    def _figure(self, monkeypatch):
        """The figure pairs draws, kept open."""
        import dftk.commands.pairs_cmd as pairs_cmd

        figs = []
        monkeypatch.setattr(
            pairs_cmd, "save_or_show", lambda fig, args: figs.append(fig)
        )
        return figs

    def test_constant_and_empty_columns(self, tmp_path, df, monkeypatch):
        figs = self._figure(monkeypatch)
        PairsCommand().execute(_args(self._tsv(tmp_path, df.assign(b=5.0, c=np.nan))))
        fig = figs[0]
        # The constant column's points are inside its axis range
        lo, hi = fig.axes[3].get_ylim()  # row b, column a
        assert lo < 5.0 < hi
        plt.close("all")

    def test_upper_corr_per_group(self, tmp_path, df, monkeypatch):
        texts = self._texts(monkeypatch)
        self._run(tmp_path, df, "-c", "a", "b", "-g", "grp", "--upper", "corr")
        x, y = df[df.grp == "x"], df[df.grp == "y"]
        assert texts == [  # first-seen group order: the TSV has no categories
            f"all: {df['a'].corr(df['b']):.2f}",
            f"y: {y['a'].corr(y['b']):.2f}",
            f"x: {x['a'].corr(x['b']):.2f}",
        ]
        plt.close("all")

    def test_group_kdes_scaled_by_share(self, tmp_path, df, monkeypatch):
        figs = self._figure(monkeypatch)
        df = pd.concat([df, df[df.grp == "x"]])  # x has 2/3 of the rows
        tsv = self._tsv(tmp_path, df)
        PairsCommand().execute(_args(tsv, "-c", "a", "b", "-g", "grp"))
        lines = figs[0].axes[0].get_lines()  # y first, as first seen
        areas = [np.trapezoid(ln.get_ydata(), ln.get_xdata()) for ln in lines]
        assert areas == pytest.approx([1 / 3, 2 / 3], abs=0.01)
        plt.close("all")

    def test_upper_corr_writes_r(self, tmp_path, df, monkeypatch):
        texts = self._texts(monkeypatch)
        self._run(tmp_path, df, "-c", "a", "b", "c", "--upper", "corr")
        expected = [
            f"{df['a'].corr(df['b']):.2f}",
            f"{df['a'].corr(df['c']):.2f}",
            f"{df['b'].corr(df['c']):.2f}",
        ]
        assert texts == expected
        plt.close("all")
