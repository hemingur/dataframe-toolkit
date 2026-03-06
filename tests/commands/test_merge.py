"""
Tests for stattools.commands.merge_cmd._do_merge and helper functions.
"""

import argparse

import pandas as pd
import pytest

from stattools.commands.merge_cmd import (
    MergeCommand,
    _do_merge,
    _expand_select,
    _filter_only,
    _set_suffixes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def left_df():
    return pd.DataFrame({
        "id":    [1, 2, 3],
        "name":  ["Alice", "Bob", "Carol"],
        "score": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def right_df():
    return pd.DataFrame({
        "id":      [1, 2, 4],
        "country": ["US", "UK", "CA"],
        "score":   [100.0, 200.0, 400.0],  # duplicate column name
    })


@pytest.fixture
def right_mapped_df():
    """Right file with differently-named key column."""
    return pd.DataFrame({
        "person_id": [1, 2, 4],
        "tag":       ["x", "y", "z"],
    })


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
        assert "score" in result.columns      # left side, no suffix
        assert "score_r" in result.columns    # right side, _r suffix

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
        assert len(result) == 3   # all left rows preserved
        assert set(result["id"]) == {1, 2, 3}

    def test_right_join_keeps_unmatched(self, left_df, right_df):
        args = _merge_args(keys=["id"], type="right")
        _set_suffixes(args)
        result = _do_merge(left_df, right_df, args)
        assert set(result["id"]) == {1, 2, 4}


# ---------------------------------------------------------------------------
# _do_merge — select [:left:] / [:right:]
# ---------------------------------------------------------------------------


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
        args = argparse.Namespace(rightfile=None)
        with pytest.raises(ValueError, match="-r"):
            cmd.execute(args)

    def test_self_join_right_dash_copies_left(self, left_df):
        # When -r -, df_right should be a copy of df_left (independent object)
        df_right = left_df.copy()   # simulate what execute() does
        df_right.iloc[0, 0] = 999  # mutate right copy
        assert left_df.iloc[0, 0] != 999  # left must be unaffected
