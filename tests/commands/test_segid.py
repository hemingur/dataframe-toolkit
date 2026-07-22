"""Tests for dftk.commands.segid_cmd."""

import io as _io
import sys

import pandas as pd
import pytest

from dftk.commands.segid_cmd import SegidCommand, _segid
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(col="col", destcol="segid", ignore=None)
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run SegidCommand on *df* and return the parsed TSV output."""
    import dftk.commands.segid_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        SegidCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


# ---------------------------------------------------------------------------
# _segid() — direct function tests
# ---------------------------------------------------------------------------


class TestSegidFunction:
    def test_string_column_runs(self):
        df = pd.DataFrame({"col": ["A", "A", "B", "B", "A"]})
        result = _segid(df, _make_args())
        assert list(result["segid"]) == [1, 1, 2, 2, 3]

    def test_numeric_column_runs(self):
        df = pd.DataFrame({"col": [1, 1, 2, 2, 1]})
        result = _segid(df, _make_args())
        assert list(result["segid"]) == [1, 1, 2, 2, 3]

    def test_all_same_value_single_segment(self):
        df = pd.DataFrame({"col": ["A", "A", "A"]})
        result = _segid(df, _make_args())
        assert list(result["segid"]) == [1, 1, 1]

    def test_all_distinct_values_each_own_segment(self):
        df = pd.DataFrame({"col": ["A", "B", "C"]})
        result = _segid(df, _make_args())
        assert list(result["segid"]) == [1, 2, 3]

    def test_ignore_value_gets_zero(self):
        df = pd.DataFrame({"col": ["A", "X", "A", "B"]})
        result = _segid(df, _make_args(ignore="X"))
        assert list(result["segid"]) == [1, 0, 1, 2]

    def test_ignore_run_continues_across_gap(self):
        # Without --ignore, the second 'A' would start a new segment (3
        # values changed: A -> X -> A). With --ignore X, X is skipped
        # entirely so the surrounding A's are treated as one run.
        df = pd.DataFrame({"col": ["A", "X", "X", "A", "B"]})
        result = _segid(df, _make_args(ignore="X"))
        assert list(result["segid"]) == [1, 0, 0, 1, 2]

    def test_no_ignore_by_default(self):
        df = pd.DataFrame({"col": [1, 1, 0, 2]})
        result = _segid(df, _make_args())
        # every value change counts, including transitions through 0
        assert list(result["segid"]) == [1, 1, 2, 3]

    def test_ignore_numeric_value(self):
        df = pd.DataFrame({"col": [1, 1, 0, 2, 2]})
        result = _segid(df, _make_args(ignore="0"))
        assert list(result["segid"]) == [1, 1, 0, 2, 2]

    def test_custom_destcol(self):
        df = pd.DataFrame({"col": ["A", "B"]})
        result = _segid(df, _make_args(destcol="run_id"))
        assert "run_id" in result.columns
        assert "segid" not in result.columns

    def test_single_row(self):
        df = pd.DataFrame({"col": ["A"]})
        result = _segid(df, _make_args())
        assert list(result["segid"]) == [1]

    def test_ignore_value_never_present(self):
        df = pd.DataFrame({"col": ["A", "A", "B"]})
        result = _segid(df, _make_args(ignore="Z"))
        assert list(result["segid"]) == [1, 1, 2]


# ---------------------------------------------------------------------------
# Command integration
# ---------------------------------------------------------------------------


class TestSegidCommand:
    def test_name_and_help(self):
        cmd = SegidCommand()
        assert cmd.name == "segid"
        assert isinstance(cmd.help, str) and cmd.help

    def test_output_schema(self):
        df = pd.DataFrame({"col": ["A", "A", "B"]})
        result = _run(df)
        assert list(result.columns) == ["col", "segid"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"col": ["A"]})
        with pytest.raises(ValueError, match="Column\\(s\\) not found"):
            _run(df, col="nope")

    def test_preserves_other_columns(self):
        df = pd.DataFrame({"col": ["A", "A", "B"], "extra": [1, 2, 3]})
        result = _run(df)
        assert list(result["extra"]) == [1, 2, 3]
