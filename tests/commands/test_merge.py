"""
Tests for dftk.commands.merge_cmd._do_merge and helper functions.
"""

import argparse
import io as _io
from types import SimpleNamespace

import pandas as pd
import pytest

from dftk.commands.merge_cmd import (
    MergeCommand,
    _do_merge,
    _expand_select,
    _set_suffixes,
)
from dftk.common import docformats
from tests.conftest import make_args

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def left_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "score": [10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def right_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 4],
            "country": ["US", "UK", "CA"],
            "score": [100.0, 200.0, 400.0],  # duplicate column name
        }
    )


@pytest.fixture
def right_mapped_df():
    """Right file with differently-named key column."""
    return pd.DataFrame(
        {
            "person_id": [1, 2, 4],
            "tag": ["x", "y", "z"],
        }
    )


def _merge_args(**overrides):
    """Build a minimal args Namespace for _do_merge."""
    defaults = dict(
        keys=None,
        left_on=None,
        right_on=None,
        type="inner",
        only=None,
        # io.printdf fields — not used in _do_merge but needed by _set_suffixes
        select=None,
        drop=None,
    )
    defaults.update(overrides)
    ns = argparse.Namespace(**defaults)
    # _set_suffixes expects these to be set
    _set_suffixes(ns)
    return ns


# ---------------------------------------------------------------------------
# _set_suffixes
# ---------------------------------------------------------------------------


class TestSetSuffixes:
    def test_default_suffixes(self):
        args = argparse.Namespace(type="inner", only=None)
        _set_suffixes(args)
        assert args.suffixes == ("", "_r")
        assert args.indicator is False

    def test_only_left_forces_outer(self):
        args = argparse.Namespace(type="inner", only="left")
        _set_suffixes(args)
        assert args.type == "outer"
        assert args.indicator is True
        assert args.filtertag == "left_only"
        assert args.suffixes == ("", "_r")

    def test_only_right_swaps_suffixes(self):
        args = argparse.Namespace(type="inner", only="right")
        _set_suffixes(args)
        assert args.suffixes == ("_l", "")
        assert args.filtertag == "right_only"

    def test_cross_clears_only(self):
        args = argparse.Namespace(type="cross", only="left")
        _set_suffixes(args)
        assert args.only is None
        assert args.indicator is False


# ---------------------------------------------------------------------------
# _expand_select
# ---------------------------------------------------------------------------


class TestExpandSelect:
    def _make_args(self, select):
        args = argparse.Namespace(select=select, suffixes=("", "_r"))
        return args

    def test_no_token_returns_default_suffixes(self):
        args = self._make_args(["id", "name"])
        result = _expand_select(["id", "name"], ["id", "country"], args)
        assert result == ("", "_r")
        assert args.select == ["id", "name"]  # unchanged

    def test_none_select_returns_default(self):
        args = self._make_args(None)
        result = _expand_select(["id"], ["id"], args)
        assert result == ("", "_r")

    def test_left_token_expanded(self):
        args = self._make_args(["[:left:]"])
        suffixes = _expand_select(["id", "name"], ["id", "country"], args)
        assert args.select == ["id", "name"]
        assert suffixes == ("", "_r")

    def test_right_token_expanded(self):
        args = self._make_args(["[:right:]"])
        suffixes = _expand_select(["id", "name"], ["id", "country"], args)
        assert args.select == ["id", "country"]
        assert suffixes == ("_l", "")

    def test_both_tokens_raises(self):
        args = self._make_args(["[:left:]", "[:right:]"])
        with pytest.raises(ValueError, match="Cannot use both"):
            _expand_select(["id"], ["id"], args)

    def test_left_token_mixed_with_other_cols(self):
        args = self._make_args(["extra", "[:left:]"])
        _expand_select(["id", "name"], ["id"], args)
        assert args.select == ["extra", "id", "name"]


# ---------------------------------------------------------------------------
# _do_merge — inner join
# ---------------------------------------------------------------------------


class TestDoMergeInner:
    def test_common_key_inner_join(self, left_df, right_df):
        args = _merge_args(keys=["id"])
        result = _do_merge(left_df, right_df, args)
        # Only ids 1 and 2 match
        assert set(result["id"]) == {1, 2}
        assert len(result) == 2

    def test_duplicate_col_gets_r_suffix(self, left_df, right_df):
        args = _merge_args(keys=["id"])
        result = _do_merge(left_df, right_df, args)
        assert "score" in result.columns  # left side, no suffix
        assert "score_r" in result.columns  # right side, _r suffix

    def test_mapped_key_merge(self, left_df, right_mapped_df):
        args = _merge_args(left_on=["id"], right_on=["person_id"])
        result = _do_merge(left_df, right_mapped_df, args)
        assert set(result["id"]) == {1, 2}
        assert "tag" in result.columns

    def test_no_keys_raises(self, left_df, right_df):
        args = _merge_args()  # keys=None, left_on=None, right_on=None
        with pytest.raises(ValueError, match="Specify -k"):
            _do_merge(left_df, right_df, args)

    def test_left_join_keeps_unmatched(self, left_df, right_df):
        args = _merge_args(keys=["id"], type="left")
        _set_suffixes(args)
        result = _do_merge(left_df, right_df, args)
        assert len(result) == 3  # all left rows preserved
        assert set(result["id"]) == {1, 2, 3}

    def test_right_join_keeps_unmatched(self, left_df, right_df):
        args = _merge_args(keys=["id"], type="right")
        _set_suffixes(args)
        result = _do_merge(left_df, right_df, args)
        assert set(result["id"]) == {1, 2, 4}


# ---------------------------------------------------------------------------
# _do_merge — select [:left:] / [:right:]
# ---------------------------------------------------------------------------


class TestJsonColumns:
    """DuckDB JSON-text columns stay marked through a merge (docformats)."""

    def test_both_sides_marked_with_suffix(self, left_df, right_df):
        docformats.mark_json_columns(left_df, ["name"])
        docformats.mark_json_columns(right_df, ["score", "country"])
        result = _do_merge(left_df, right_df, _merge_args(keys=["id"]))
        assert docformats.json_columns(result) == ["name", "score_r", "country"]

    def test_unmarked_inputs(self, left_df, right_df):
        result = _do_merge(left_df, right_df, _merge_args(keys=["id"]))
        assert docformats.json_columns(result) == []


class TestDoMergeSelectToken:
    def test_select_left_expands_args_select(self, left_df, right_df):
        # [:left:] is expanded in-place on args.select before the merge.
        # _do_merge returns the full merged df; io.printdf applies the select.
        args = _merge_args(keys=["id"], select=["[:left:]"])
        _do_merge(left_df, right_df, args)
        # args.select now contains the actual left column names
        assert args.select == list(left_df.columns)
        # The token is gone
        assert "[:left:]" not in args.select
        # score_r would appear in the df but not in args.select → io.printdf drops it
        assert "score_r" not in args.select

    def test_select_right_expands_args_select(self, left_df, right_df):
        args = _merge_args(keys=["id"], select=["[:right:]"])
        _do_merge(left_df, right_df, args)
        # args.select now contains the actual right column names
        assert args.select == list(right_df.columns)
        assert "[:right:]" not in args.select
        # score_l would appear in the df but not in args.select
        assert "score_l" not in args.select


# ---------------------------------------------------------------------------
# _do_merge — --only (anti-join)
# ---------------------------------------------------------------------------


class TestDoMergeOnly:
    def test_only_left_returns_unmatched_left_rows(self, left_df, right_df):
        args = _merge_args(keys=["id"], only="left")
        result = _do_merge(left_df, right_df, args)
        # id=3 is only in left
        assert set(result["id"]) == {3}

    def test_only_right_returns_unmatched_right_rows(self, left_df, right_df):
        args = _merge_args(keys=["id"], only="right")
        result = _do_merge(left_df, right_df, args)
        # id=4 is only in right
        assert set(result["id"]) == {4}

    def test_only_left_no_merge_indicator_column(self, left_df, right_df):
        args = _merge_args(keys=["id"], only="left")
        result = _do_merge(left_df, right_df, args)
        assert "_merge" not in result.columns

    def test_only_left_default_select_is_left_cols(self, left_df, right_df):
        # When --select is not given, _filter_only sets args.select to left columns.
        # io.printdf will then filter out right-only columns like "country".
        args = _merge_args(keys=["id"], only="left")
        _do_merge(left_df, right_df, args)
        # args.select should be set to left df columns (no right-side extras)
        assert args.select is not None
        assert "country" not in args.select
        assert "name" in args.select


# ---------------------------------------------------------------------------
# _do_merge — cross join
# ---------------------------------------------------------------------------


class TestDoMergeCross:
    def test_cross_join_row_count(self, left_df, right_df):
        # 3 left rows × 3 right rows = 9
        args = _merge_args(type="cross")
        _set_suffixes(args)
        result = _do_merge(left_df, right_df, args)
        assert len(result) == 9

    def test_cross_join_no_uuid_col(self, left_df, right_df):
        # The temporary UUID key column should be in args.drop and removed by printdf,
        # but _do_merge itself doesn't call printdf — so we check args.drop was set.
        args = _merge_args(type="cross")
        _set_suffixes(args)
        _do_merge(left_df, right_df, args)
        # args.drop should contain the UUID col name
        assert args.drop is not None and len(args.drop) == 1

    def test_cross_join_clears_only(self):
        # _set_suffixes should clear --only for cross join
        args = argparse.Namespace(type="cross", only="left")
        _set_suffixes(args)
        assert args.only is None


# ---------------------------------------------------------------------------
# Outer join
# ---------------------------------------------------------------------------


class TestDoMergeOuter:
    def test_outer_join_all_ids(self, left_df, right_df):
        args = _merge_args(keys=["id"], type="outer")
        _set_suffixes(args)
        result = _do_merge(left_df, right_df, args)
        assert set(result["id"]) == {1, 2, 3, 4}

    def test_outer_join_no_merge_col(self, left_df, right_df):
        # Without --only the _merge indicator is not requested, so absent
        args = _merge_args(keys=["id"], type="outer")
        result = _do_merge(left_df, right_df, args)
        assert "_merge" not in result.columns


# ---------------------------------------------------------------------------
# Self-join (-r - / -l - -r -)
# ---------------------------------------------------------------------------


class TestSelfJoin:
    def test_self_join_inner_returns_all_rows(self, left_df):
        # Joining a df with itself on a unique key returns the same rows
        args = _merge_args(keys=["id"])
        result = _do_merge(left_df, left_df.copy(), args)
        assert len(result) == len(left_df)
        assert set(result["id"]) == set(left_df["id"])

    def test_self_join_duplicate_col_gets_r_suffix(self, left_df):
        args = _merge_args(keys=["id"])
        result = _do_merge(left_df, left_df.copy(), args)
        # name and score from right get _r suffix
        assert "name_r" in result.columns
        assert "score_r" in result.columns

    def test_self_join_only_left_returns_nothing_for_unique_key(self, left_df):
        # Every row matches itself → no row is exclusive to one side
        args = _merge_args(keys=["id"], only="left")
        result = _do_merge(left_df, left_df.copy(), args)
        assert len(result) == 0

    def test_self_join_no_rightfile_raises(self):
        # execute() must raise when --rightfile is not provided
        cmd = MergeCommand()
        args = argparse.Namespace(rightfile=None, leftfile=None, DATAFILE=None)
        with pytest.raises(ValueError, match="-r"):
            cmd.execute(args)

    def test_self_join_right_dash_copies_left(self, left_df):
        # When -r -, df_right should be a copy of df_left (independent object)
        df_right = left_df.copy()  # simulate what execute() does
        df_right.iloc[0, 0] = 999  # mutate right copy
        assert left_df.iloc[0, 0] != 999  # left must be unaffected


# ---------------------------------------------------------------------------
# Which side reads stdin (execute())
# ---------------------------------------------------------------------------


class TestStdinSides:
    """Either side can come from stdin; both from stdin is a self-join."""

    def _reads(self, monkeypatch, tty=False, **kwargs):
        """Run execute() and return the DATAFILE of each io.read call."""
        import dftk.commands.merge_cmd as mod

        reads = []

        def fake_read(args):
            reads.append(args.DATAFILE)
            return pd.DataFrame({"id": [1, 2], "val": [10.0, 20.0]})

        monkeypatch.setattr(mod.io, "read", fake_read)
        monkeypatch.setattr(mod.io, "printdf", lambda df, args: None)
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: tty))
        defaults = dict(
            keys=["id"],
            left_on=None,
            right_on=None,
            type="inner",
            only=None,
            leftfile=None,
            rightfile=None,
            DATAFILE=None,
        )
        defaults.update(kwargs)
        mod.MergeCommand().execute(make_args(**defaults))
        return reads

    @pytest.mark.parametrize("left", [None, "-", "..."])
    def test_both_stdin_is_a_self_join(self, monkeypatch, left):
        assert self._reads(monkeypatch, leftfile=left, rightfile="-") == [left]

    def test_positional_stdin_with_dash_right_is_a_self_join(self, monkeypatch):
        assert self._reads(monkeypatch, DATAFILE="-", rightfile="-") == ["-"]

    def test_stdin_left_named_right(self, monkeypatch):
        reads = self._reads(monkeypatch, leftfile="-", rightfile="r.tsv")
        assert reads == ["-", "r.tsv"]

    @pytest.mark.parametrize("right", ["-", "..."])
    def test_named_left_dash_right_reads_stdin(self, monkeypatch, right):
        reads = self._reads(monkeypatch, leftfile="left.tsv", rightfile=right)
        assert reads == ["left.tsv", right]

    def test_named_left_no_right_reads_stdin(self, monkeypatch):
        assert self._reads(monkeypatch, leftfile="left.tsv") == ["left.tsv", "-"]

    def test_positional_left_no_right_reads_stdin(self, monkeypatch):
        assert self._reads(monkeypatch, DATAFILE="left.tsv") == ["left.tsv", "-"]

    def test_leftfile_overrides_positional(self, monkeypatch):
        reads = self._reads(monkeypatch, DATAFILE="x.tsv", leftfile="left.tsv")
        assert reads == ["left.tsv", "-"]

    def test_no_right_and_stdin_left_raises(self, monkeypatch):
        with pytest.raises(ValueError, match="-r FILE"):
            self._reads(monkeypatch, leftfile="-")

    def test_no_right_and_nothing_piped_raises(self, monkeypatch):
        # A terminal on stdin: don't sit waiting for input
        with pytest.raises(ValueError, match="-r FILE"):
            self._reads(monkeypatch, tty=True, leftfile="left.tsv")


class TestStdinParquetPath:
    """End to end through io.read: stdin holds the path printed by -o."""

    @staticmethod
    def _run(monkeypatch, tmp_path, capsys, stdin_df, **kwargs):
        from dftk.common.io import _write_parquet

        piped = tmp_path / "piped.parquet"
        _write_parquet(stdin_df, str(piped))
        monkeypatch.setattr(
            "sys.stdin",
            SimpleNamespace(
                buffer=_io.BytesIO(f"{piped}\n".encode()),
                isatty=lambda: False,
            ),
        )
        defaults = dict(
            keys=["id"],
            left_on=None,
            right_on=None,
            type="inner",
            only=None,
            leftfile=None,
            rightfile=None,
            DATAFILE=None,
            backend="pandas",
            output=None,
        )
        defaults.update(kwargs)
        MergeCommand().execute(make_args(**defaults))
        return pd.read_csv(_io.StringIO(capsys.readouterr().out), sep="\t")

    def test_right_side_from_stdin(self, monkeypatch, tmp_path, capsys):
        left = tmp_path / "left.tsv"
        left.write_text("id\ta\n1\tx\n2\ty\n")
        stdin_df = pd.DataFrame({"id": [2, 1], "b": ["q", "p"]})
        out = self._run(
            monkeypatch, tmp_path, capsys, stdin_df, leftfile=str(left), rightfile="-"
        )
        assert out.to_dict("list") == {"id": [1, 2], "a": ["x", "y"], "b": ["p", "q"]}

    def test_self_join_from_stdin(self, monkeypatch, tmp_path, capsys):
        stdin_df = pd.DataFrame({"id": [1, 2], "a": ["x", "y"]})
        out = self._run(
            monkeypatch, tmp_path, capsys, stdin_df, leftfile="-", rightfile="-"
        )
        assert out.to_dict("list") == {
            "id": [1, 2],
            "a": ["x", "y"],
            "a_r": ["x", "y"],
        }


# ---------------------------------------------------------------------------
# Integer columns in unmatched rows
# ---------------------------------------------------------------------------


class TestIntegerGaps:
    """Joins that leave gaps keep integer columns as nullable integers."""

    @pytest.fixture
    def left(self):
        return pd.DataFrame({"id": [1, 2, 3], "age": [34, 51, 47]})

    @pytest.fixture
    def right(self):
        return pd.DataFrame({"id": [1, 3, 9], "depth": [100, 300, 900]})

    def test_left_join_right_column_nullable(self, left, right):
        result = _do_merge(left, right, _merge_args(keys=["id"], type="left"))
        assert str(result["depth"].dtype) == "Int64"
        assert result["depth"].tolist() == [100, pd.NA, 300]
        assert str(result["age"].dtype) == "int64"  # no gaps: unchanged
        assert str(result["id"].dtype) == "int64"

    def test_right_join_left_column_nullable(self, left, right):
        result = _do_merge(left, right, _merge_args(keys=["id"], type="right"))
        assert str(result["age"].dtype) == "Int64"
        assert str(result["depth"].dtype) == "int64"
        assert result["age"].tolist() == [34, 47, pd.NA]

    def test_outer_join_both_sides_nullable(self, left, right):
        result = _do_merge(left, right, _merge_args(keys=["id"], type="outer"))
        assert str(result["age"].dtype) == "Int64"
        assert str(result["depth"].dtype) == "Int64"
        assert str(result["id"].dtype) == "int64"  # keys come from both sides
        assert result["id"].tolist() == [1, 2, 3, 9]

    def test_inner_join_unchanged(self, left, right):
        result = _do_merge(left, right, _merge_args(keys=["id"]))
        assert str(result["age"].dtype) == "int64"
        assert str(result["depth"].dtype) == "int64"

    def test_only_left_keeps_integers(self, left, right):
        # --only runs an outer join first, which used to make age float64
        result = _do_merge(left, right, _merge_args(keys=["id"], only="left"))
        assert pd.api.types.is_integer_dtype(result["age"])
        assert result["age"].tolist() == [51]

    def test_mapped_key_on_gap_side_nullable(self, left):
        right = pd.DataFrame({"person": [1, 3], "depth": [100, 300]})
        args = _merge_args(left_on=["id"], right_on=["person"], type="left")
        result = _do_merge(left, right, args)
        assert str(result["person"].dtype) == "Int64"
        assert result["person"].tolist() == [1, pd.NA, 3]

    def test_smaller_and_unsigned_ints(self, left):
        right = pd.DataFrame(
            {
                "id": [1],
                "a": pd.Series([5], dtype="int32"),
                "b": pd.Series([7], dtype="uint8"),
            }
        )
        result = _do_merge(left, right, _merge_args(keys=["id"], type="left"))
        assert str(result["a"].dtype) == "Int32"
        assert str(result["b"].dtype) == "UInt8"

    def test_inputs_not_modified(self, left, right):
        _do_merge(left, right, _merge_args(keys=["id"], type="outer"))
        assert str(left["age"].dtype) == "int64"
        assert str(right["depth"].dtype) == "int64"

    def test_nullable_input_stays_nullable(self, left):
        right = pd.DataFrame(
            {"id": [1, 2], "depth": pd.array([100, None], dtype="Int64")}
        )
        result = _do_merge(left, right, _merge_args(keys=["id"], type="left"))
        assert str(result["depth"].dtype) == "Int64"
        assert result["depth"].tolist() == [100, pd.NA, pd.NA]
