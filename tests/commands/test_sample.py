"""Tests for stattools.commands.sample_cmd."""

import io as _io
import sys

import pandas as pd

from stattools.commands.sample_cmd import SampleCommand
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(
        samplesize=None, samplefrac=None, groupcol=None, randomseed=None, replace=False
    )
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    import stattools.commands.sample_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        SampleCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSampleN:
    def test_exact_count(self):
        df = pd.DataFrame({"x": range(100)})
        result = _run(df, samplesize=10)
        assert len(result) == 10

    def test_rows_are_subset(self):
        df = pd.DataFrame({"x": range(100)})
        result = _run(df, samplesize=20)
        assert set(result["x"]).issubset(set(df["x"]))

    def test_no_duplicates_without_replace(self):
        df = pd.DataFrame({"x": range(50)})
        result = _run(df, samplesize=50)
        assert len(result) == len(result.drop_duplicates())

    def test_with_replacement_allows_duplicates(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        # Sample 100 from 3 with replacement — duplicates are certain
        result = _run(df, samplesize=100, replace=True)
        assert len(result) == 100


class TestSampleFrac:
    def test_half(self):
        df = pd.DataFrame({"x": range(100)})
        result = _run(df, samplefrac=0.5)
        assert len(result) == 50

    def test_full_shuffle(self):
        df = pd.DataFrame({"x": range(20)})
        result = _run(df, samplefrac=1.0)
        assert len(result) == 20
        assert set(result["x"]) == set(df["x"])


class TestSampleSeed:
    def test_integer_seed_reproducible(self):
        df = pd.DataFrame({"x": range(100)})
        r1 = _run(df, samplesize=10, randomseed="42")
        r2 = _run(df, samplesize=10, randomseed="42")
        assert list(r1["x"]) == list(r2["x"])

    def test_different_seeds_differ(self):
        df = pd.DataFrame({"x": range(100)})
        r1 = _run(df, samplesize=10, randomseed="1")
        r2 = _run(df, samplesize=10, randomseed="2")
        assert list(r1["x"]) != list(r2["x"])

    def test_string_seed_reproducible(self):
        df = pd.DataFrame({"x": range(100)})
        r1 = _run(df, samplesize=10, randomseed="hello")
        r2 = _run(df, samplesize=10, randomseed="hello")
        assert list(r1["x"]) == list(r2["x"])

    def test_seed_printed_to_stderr(self, capsys):
        df = pd.DataFrame({"x": range(10)})
        _run(df, samplesize=5, randomseed="42")
        assert "42" in capsys.readouterr().err

    def test_no_seed_no_stderr(self, capsys):
        df = pd.DataFrame({"x": range(10)})
        _run(df, samplesize=5)
        assert capsys.readouterr().err == ""


class TestSampleGroupcol:
    def test_samples_within_groups(self):
        df = pd.DataFrame(
            {
                "group": ["a"] * 20 + ["b"] * 20,
                "x": range(40),
            }
        )
        result = _run(df, samplesize=5, groupcol=["group"])
        counts = result["group"].value_counts()
        assert counts["a"] == 5
        assert counts["b"] == 5

    def test_group_columns_preserved(self):
        df = pd.DataFrame(
            {
                "g": ["x", "x", "y", "y"],
                "v": [1, 2, 3, 4],
            }
        )
        result = _run(df, samplesize=1, groupcol=["g"])
        assert "g" in result.columns
        assert "v" in result.columns

    def test_frac_within_groups(self):
        df = pd.DataFrame(
            {
                "group": ["a"] * 10 + ["b"] * 10,
                "x": range(20),
            }
        )
        result = _run(df, samplefrac=0.5, groupcol=["group"])
        counts = result["group"].value_counts()
        assert counts["a"] == 5
        assert counts["b"] == 5


class TestSampleColumns:
    def test_all_columns_preserved(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        result = _run(df, samplesize=2)
        assert set(result.columns) == {"a", "b", "c"}

    def test_index_reset(self):
        df = pd.DataFrame({"x": range(10)})
        result = _run(df, samplesize=5)
        assert list(result.index) == list(range(5))
