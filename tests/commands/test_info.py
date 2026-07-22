"""Tests for dftk.commands.info_cmd."""

import io as _io
import sys

import numpy as np
import pandas as pd

from dftk.commands.info_cmd import InfoCommand, _info, _summary_line
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(summary=False)
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run InfoCommand on *df* and return the parsed TSV output."""
    import dftk.commands.info_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        InfoCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


# ---------------------------------------------------------------------------
# _info()
# ---------------------------------------------------------------------------


class TestInfoFunction:
    def test_column_order_preserved(self):
        df = pd.DataFrame({"b": [1], "a": [2], "c": [3]})
        result = _info(df)
        assert list(result["name"]) == ["b", "a", "c"]

    def test_no_nulls(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        row = _info(df).iloc[0]
        assert row["non_null"] == 3
        assert row["null"] == 0

    def test_nulls_counted(self):
        df = pd.DataFrame({"x": [1.0, np.nan, np.nan]})
        row = _info(df).iloc[0]
        assert row["non_null"] == 1
        assert row["null"] == 2

    def test_dtype_reported(self):
        df = pd.DataFrame({"i": [1, 2], "f": [1.0, 2.0], "b": [True, False]})
        result = _info(df)
        dtypes = dict(zip(result["name"], result["dtype"], strict=True))
        assert "int" in dtypes["i"]
        assert "float" in dtypes["f"]
        assert "bool" in dtypes["b"]

    def test_memory_bytes_positive(self):
        df = pd.DataFrame({"x": range(100)})
        row = _info(df).iloc[0]
        assert row["memory_bytes"] > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = _info(df)
        assert len(result) == 1
        assert result.iloc[0]["non_null"] == 0
        assert result.iloc[0]["null"] == 0

    def test_no_columns(self):
        df = pd.DataFrame(index=[0, 1, 2])
        result = _info(df)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _summary_line()
# ---------------------------------------------------------------------------


class TestSummaryLine:
    def test_shape_reported(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        line = _summary_line(df)
        assert "3 rows" in line
        assert "2 columns" in line

    def test_no_duplicates(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert "0 duplicated" in _summary_line(df)

    def test_duplicates_counted(self):
        df = pd.DataFrame({"x": [1, 1, 1, 2]})
        assert "2 duplicated" in _summary_line(df)


# ---------------------------------------------------------------------------
# Command integration
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_name_and_help(self):
        cmd = InfoCommand()
        assert cmd.name == "info"
        assert isinstance(cmd.help, str) and cmd.help

    def test_output_schema(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = _run(df)
        assert list(result.columns) == [
            "name",
            "dtype",
            "non_null",
            "null",
            "memory_bytes",
        ]
        assert len(result) == 2

    def test_summary_goes_to_stderr(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _run(df, summary=True)
        captured = capsys.readouterr()
        assert "rows" in captured.err
        assert "columns" in captured.err

    def test_no_stderr_without_summary(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _run(df)
        assert capsys.readouterr().err == ""

    def test_summary_reflects_duplicates(self, capsys):
        df = pd.DataFrame({"x": [1, 1, 2]})
        _run(df, summary=True)
        captured = capsys.readouterr()
        assert "1 duplicated" in captured.err
