"""
Tests for the check_cols() validation utility and the per-command
column-existence checks introduced in stat, pivot, and merge.
"""

import argparse
import logging

import pandas as pd
import pytest

from stattools.common.io import check_cols, io


# ---------------------------------------------------------------------------
# check_cols() unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})


class TestCheckCols:

    def test_valid_cols_no_error(self, sample_df):
        check_cols(sample_df, ["a", "b"])   # should not raise

    def test_missing_col_raises(self, sample_df):
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            check_cols(sample_df, ["a", "z"])

    def test_error_shows_missing_name(self, sample_df):
        with pytest.raises(ValueError, match="z"):
            check_cols(sample_df, ["z"])

    def test_error_shows_available_cols(self, sample_df):
        with pytest.raises(ValueError, match="a"):
            check_cols(sample_df, ["z"])   # message lists available cols

    def test_context_label_in_error(self, sample_df):
        with pytest.raises(ValueError, match=r"-c/--cols"):
            check_cols(sample_df, ["z"], "-c/--cols")

    def test_none_cols_is_no_op(self, sample_df):
        check_cols(sample_df, None)         # should not raise

    def test_empty_list_is_no_op(self, sample_df):
        check_cols(sample_df, [])           # should not raise

    def test_multiple_missing_all_reported(self, sample_df):
        with pytest.raises(ValueError) as exc_info:
            check_cols(sample_df, ["x", "y", "z"])
        msg = str(exc_info.value)
        assert "x" in msg
        assert "y" in msg
        assert "z" in msg


# ---------------------------------------------------------------------------
# stat command — column validation
# ---------------------------------------------------------------------------


class TestStatColumnValidation:

    def test_missing_col_raises(self):
        from stattools.commands.stat_cmd import StatCommand
        cmd = StatCommand()
        df = pd.DataFrame({"x": [1, 2, 3]})
        args = argparse.Namespace(
            cols=["nonexistent"],
            groupcol=None,
            bootstrap=None,
            randomseed=None,
            replace=True,
            samplesize=None,
            samplefrac=1.0,
            confidencelevel=95.0,
            confidencemethod="linear",
            DATAFILE=None,
        )
        with pytest.raises(ValueError, match="nonexistent"):
            from stattools.common.io import check_cols
            check_cols(df, args.cols, "-c/--cols")

    def test_missing_groupcol_raises(self):
        df = pd.DataFrame({"x": [1, 2, 3], "g": ["A", "A", "B"]})
        with pytest.raises(ValueError, match="bad_group"):
            check_cols(df, ["bad_group"], "-g/--groupcol")

    def test_valid_cols_no_error(self):
        df = pd.DataFrame({"x": [1, 2, 3], "g": ["A", "A", "B"]})
        check_cols(df, ["x"], "-c/--cols")
        check_cols(df, ["g"], "-g/--groupcol")


# ---------------------------------------------------------------------------
# pivot command — column validation
# ---------------------------------------------------------------------------


class TestPivotColumnValidation:

    def test_missing_value_col_raises(self):
        df = pd.DataFrame({"x": [1, 2], "group": ["A", "B"]})
        with pytest.raises(ValueError, match="bad_col"):
            check_cols(df, ["bad_col"], "-v/--values")

    def test_missing_index_col_raises(self):
        df = pd.DataFrame({"x": [1, 2], "group": ["A", "B"]})
        with pytest.raises(ValueError, match="no_such_index"):
            check_cols(df, ["no_such_index"], "-i/--index")

    def test_valid_pivot_cols_no_error(self):
        df = pd.DataFrame({"x": [1, 2], "group": ["A", "B"]})
        check_cols(df, ["x"], "-v/--values")
        check_cols(df, ["group"], "-i/--index")


# ---------------------------------------------------------------------------
# merge command — column validation
# ---------------------------------------------------------------------------


class TestMergeColumnValidation:

    def test_missing_key_in_left_raises(self):
        left = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        right = pd.DataFrame({"id": [1, 2], "tag": ["a", "b"]})
        with pytest.raises(ValueError, match="bad_key"):
            check_cols(left, ["bad_key"], "-k (left file)")

    def test_missing_key_in_right_raises(self):
        left = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        right = pd.DataFrame({"other_id": [1, 2], "tag": ["a", "b"]})
        with pytest.raises(ValueError, match="id"):
            check_cols(right, ["id"], "-k (right file)")

    def test_missing_left_on_raises(self):
        left = pd.DataFrame({"person_id": [1, 2]})
        with pytest.raises(ValueError, match="no_such_col"):
            check_cols(left, ["no_such_col"], "-lo/--left_on")

    def test_valid_keys_no_error(self):
        left = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        right = pd.DataFrame({"id": [1, 2], "tag": ["a", "b"]})
        check_cols(left, ["id"], "-k (left file)")
        check_cols(right, ["id"], "-k (right file)")


# ---------------------------------------------------------------------------
# io.printdf — --select literal miss warning
# ---------------------------------------------------------------------------


class TestSelectLiteralWarning:

    def _printdf_args(self, select, **extra):
        defaults = dict(
            select=select,
            drop=None, move=None, na_rep=None, dropna=False,
            postquery=[], cast=None, sortasc=None, sortdesc=None, sort=None,
            expect=None, round=None, sigdig=None,
            movetofront=None, movetoback=None,
            deduplicate=None, noheader=False, removeheader=False,
            output=None, digits=None, errortag="-",
        )
        defaults.update(extra)
        return argparse.Namespace(**defaults)

    def test_literal_miss_emits_warning(self, sample_df, caplog):
        args = self._printdf_args(select=["a", "nonexistent"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert any("nonexistent" in r.message for r in caplog.records)

    def test_glob_miss_no_warning(self, sample_df, caplog):
        # A glob pattern that matches nothing should NOT warn
        args = self._printdf_args(select=["z*"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert not any("--select" in r.message for r in caplog.records)

    def test_valid_select_no_warning(self, sample_df, caplog):
        args = self._printdf_args(select=["a", "b"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert not any("--select" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# io.printdf — --movetofront / --movetoback missing column warning
# ---------------------------------------------------------------------------


class TestMoveToWarning:

    def _printdf_args(self, **kwargs):
        defaults = dict(
            select=None, drop=None, move=None, na_rep=None, dropna=False,
            postquery=[], cast=None, sortasc=None, sortdesc=None, sort=None,
            expect=None, round=None, sigdig=None,
            movetofront=None, movetoback=None,
            deduplicate=None, noheader=False, removeheader=False,
            output=None, digits=None, errortag="-",
        )
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_movetofront_missing_warns(self, sample_df, caplog):
        args = self._printdf_args(movetofront=["z"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert any("movetofront" in r.message and "z" in r.message
                   for r in caplog.records)

    def test_movetoback_missing_warns(self, sample_df, caplog):
        args = self._printdf_args(movetoback=["z"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert any("movetoback" in r.message and "z" in r.message
                   for r in caplog.records)

    def test_movetofront_valid_no_warning(self, sample_df, caplog):
        args = self._printdf_args(movetofront=["c"])
        with caplog.at_level(logging.WARNING):
            io.printdf(sample_df, args)
        assert not any("movetofront" in r.message for r in caplog.records)

    def test_movetofront_actually_moves(self, sample_df, caplog):
        import io as _io
        import sys
        args = self._printdf_args(movetofront=["c"])
        # Capture output to verify column order
        buf = _io.StringIO()
        args.output = None
        old_stdout = sys.stdout
        sys.stdout = buf
        io.printdf(sample_df, args)
        sys.stdout = old_stdout
        header = buf.getvalue().split("\n")[0]
        assert header.startswith("c")
