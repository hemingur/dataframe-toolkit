"""Tests for stattools.commands.concat_cmd."""

import io as _io
import sys

import pandas as pd
import pytest

from stattools.commands.concat_cmd import ConcatCommand
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(sourcecol=None, fill=None)
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(dataframes: dict, file_list: list[str], **kwargs) -> pd.DataFrame:
    """Run ConcatCommand with a mapping of filename→DataFrame."""
    import stattools.commands.concat_cmd as mod

    original_read = mod.io.read

    def _fake_read(args):
        path = args.DATAFILE
        if path not in dataframes:
            raise FileNotFoundError(path)
        return dataframes[path].copy()

    mod.io.read = _fake_read
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        args = _make_args(DATAFILES=file_list, **kwargs)
        ConcatCommand().execute(args)
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read

    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConcatBasic:
    def test_two_identical_schema(self):
        a = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        b = pd.DataFrame({"x": [5, 6], "y": [7, 8]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert len(result) == 4
        assert list(result["x"]) == [1, 2, 5, 6]

    def test_three_files(self):
        dfs = {f"{i}.tsv": pd.DataFrame({"v": [i]}) for i in range(3)}
        result = _run(dfs, ["0.tsv", "1.tsv", "2.tsv"])
        assert len(result) == 3
        assert list(result["v"]) == [0, 1, 2]

    def test_column_order_follows_first_file(self):
        a = pd.DataFrame({"x": [1], "y": [2]})
        b = pd.DataFrame({"x": [3], "y": [4]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert list(result.columns[:2]) == ["x", "y"]

    def test_index_reset(self):
        a = pd.DataFrame({"x": [1, 2]})
        b = pd.DataFrame({"x": [3, 4]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert list(result.index) == [0, 1, 2, 3]


class TestConcatMismatchedColumns:
    def test_outer_join_fills_nan(self):
        a = pd.DataFrame({"x": [1], "y": [2]})
        b = pd.DataFrame({"x": [3], "z": [4]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert "y" in result.columns
        assert "z" in result.columns
        assert pd.isna(result.loc[0, "z"])

    def test_fill_value_replaces_nan(self):
        a = pd.DataFrame({"x": [1], "y": [2]})
        b = pd.DataFrame({"x": [3]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"], fill="0")
        assert float(result.loc[1, "y"]) == 0.0

    def test_fill_string_value(self):
        a = pd.DataFrame({"x": ["a"], "tag": ["yes"]})
        b = pd.DataFrame({"x": ["b"]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"], fill="none")
        assert result.loc[1, "tag"] == "none"


class TestConcatSourceCol:
    def test_sourcecol_added(self):
        a = pd.DataFrame({"x": [1]})
        b = pd.DataFrame({"x": [2]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"], sourcecol="source")
        assert "source" in result.columns
        assert result.loc[0, "source"] == "a.tsv"
        assert result.loc[1, "source"] == "b.tsv"

    def test_sourcecol_custom_name(self):
        a = pd.DataFrame({"x": [1]})
        result = _run({"a.tsv": a}, ["a.tsv"], sourcecol="file")
        assert "file" in result.columns

    def test_no_sourcecol_by_default(self):
        a = pd.DataFrame({"x": [1]})
        b = pd.DataFrame({"x": [2]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert "source" not in result.columns


class TestConcatEdgeCases:
    def test_single_file(self):
        a = pd.DataFrame({"x": [1, 2, 3]})
        result = _run({"a.tsv": a}, ["a.tsv"])
        assert len(result) == 3

    def test_empty_file_skipped(self, tmp_path):
        import stattools.commands.concat_cmd as mod

        original_read = mod.io.read
        calls = []

        def _fake_read(args):
            calls.append(args.DATAFILE)
            if args.DATAFILE == "empty.tsv":
                raise pd.errors.EmptyDataError
            return pd.DataFrame({"x": [1]})

        mod.io.read = _fake_read
        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            args = _make_args(DATAFILES=["empty.tsv", "good.tsv"])
            ConcatCommand().execute(args)
        finally:
            sys.stdout = old_stdout
            mod.io.read = original_read

        result = pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")
        assert len(result) == 1

    def test_all_empty_raises(self):
        import stattools.commands.concat_cmd as mod

        original_read = mod.io.read
        mod.io.read = lambda args: (_ for _ in ()).throw(pd.errors.EmptyDataError)
        try:
            with pytest.raises(ValueError, match="no data"):
                args = _make_args(DATAFILES=["empty.tsv"])
                ConcatCommand().execute(args)
        finally:
            mod.io.read = original_read

    def test_duplicate_stdin_raises(self):
        with pytest.raises(ValueError, match="stdin"):
            args = _make_args(DATAFILES=["-", "..."])
            # Don't need to run fully — the check is in execute before any read
            import stattools.commands.concat_cmd as mod

            original_read = mod.io.read
            mod.io.read = lambda args: pd.DataFrame({"x": [1]})
            try:
                ConcatCommand().execute(args)
            finally:
                mod.io.read = original_read

    def test_duplicate_stdin_reversed_order_raises(self):
        """... then - should also be rejected (two stdin sources)."""
        import stattools.commands.concat_cmd as mod

        original_read = mod.io.read
        mod.io.read = lambda args: pd.DataFrame({"x": [1]})
        try:
            with pytest.raises(ValueError, match="stdin"):
                args = _make_args(DATAFILES=["...", "-"])
                ConcatCommand().execute(args)
        finally:
            mod.io.read = original_read


class TestConcatFill:
    def test_fill_float_value(self):
        """fill='0.5' should be parsed as float and applied to NaN cells."""
        a = pd.DataFrame({"x": [1.0], "y": [2.0]})
        b = pd.DataFrame({"x": [3.0]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"], fill="0.5")
        assert float(result.loc[1, "y"]) == pytest.approx(0.5)

    def test_fill_integer_coercion(self):
        """fill='1' should coerce to int 1, not the string '1'."""
        a = pd.DataFrame({"x": [1], "y": [10]})
        b = pd.DataFrame({"x": [2]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"], fill="1")
        assert float(result.loc[1, "y"]) == pytest.approx(1.0)

    def test_fill_and_sourcecol_together(self):
        """--fill and --sourcecol can be used simultaneously."""
        a = pd.DataFrame({"x": [1], "extra": [99]})
        b = pd.DataFrame({"x": [2]})
        result = _run(
            {"a.tsv": a, "b.tsv": b},
            ["a.tsv", "b.tsv"],
            fill="0",
            sourcecol="src",
        )
        assert "src" in result.columns
        # b had no 'extra' column — fill should have replaced the NaN
        assert float(result.loc[1, "extra"]) == pytest.approx(0.0)
        assert result.loc[1, "src"] == "b.tsv"

    def test_no_fill_preserves_nan(self):
        """Without --fill, missing columns stay as NaN."""
        a = pd.DataFrame({"x": [1], "y": [2]})
        b = pd.DataFrame({"x": [3]})
        result = _run({"a.tsv": a, "b.tsv": b}, ["a.tsv", "b.tsv"])
        assert pd.isna(result.loc[1, "y"])
