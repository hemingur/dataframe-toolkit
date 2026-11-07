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

    def test_glob_matching_nothing(self, df):
        with pytest.raises(ValueError, match="matches no numeric column"):
            _columns(df, ["z*", "a"], exclude=set())


class TestGroups:
    def test_categories_keep_their_order(self, df):
        assert [label for label, _ in _groups(df, "grp", None)] == ["x", "y"]

    def test_groupcolorder(self, df):
        assert [label for label, _ in _groups(df, "grp", ["y"])] == ["y", "x"]

    def test_no_group(self, df):
        assert _groups(df, None, None)[0][0] is None


class TestExecute:
    def _run(self, tmp_path, df, *argv):
        data = tmp_path / "in.tsv"
        df.to_csv(data, sep="\t", index=False)
        png = tmp_path / "out.png"
        PairsCommand().execute(_args(str(data), *argv, "-f", str(png)))
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

    def test_upper_corr_writes_r(self, tmp_path, df, monkeypatch):
        texts = []
        orig = matplotlib.axes.Axes.text
        monkeypatch.setattr(
            matplotlib.axes.Axes,
            "text",
            lambda self, x, y, s, **kw: texts.append(s) or orig(self, x, y, s, **kw),
        )
        self._run(tmp_path, df, "-c", "a", "b", "c", "--upper", "corr")
        expected = [
            f"{df['a'].corr(df['b']):.2f}",
            f"{df['a'].corr(df['c']):.2f}",
            f"{df['b'].corr(df['c']):.2f}",
        ]
        assert texts == expected
        plt.close("all")
