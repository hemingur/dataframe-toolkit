"""Tests for dftk.commands.transpose_cmd."""

import io as _io
import sys

import pandas as pd

from dftk.commands.transpose_cmd import TransposeCommand
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(keycol="column")
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run TransposeCommand on *df* and return the parsed TSV output."""
    import dftk.commands.transpose_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        TransposeCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


class TestTransposeCommand:
    def test_name_and_help(self):
        cmd = TransposeCommand()
        assert cmd.name == "transpose"
        assert isinstance(cmd.help, str) and cmd.help

    def test_shape_flipped(self):
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        result = _run(df)
        # 2 original columns -> 2 data rows; 3 original rows -> 3 + keycol columns
        assert len(result) == 2
        assert len(result.columns) == 4

    def test_default_keycol_name(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = _run(df)
        assert result.columns[0] == "column"

    def test_custom_keycol_name(self):
        df = pd.DataFrame({"x": [1, 2]})
        result = _run(df, keycol="field")
        assert result.columns[0] == "field"

    def test_column_names_become_key_values(self):
        df = pd.DataFrame({"sample": ["s1", "s2"], "x": [1, 3], "y": [2, 4]})
        result = _run(df)
        assert list(result["column"]) == ["sample", "x", "y"]

    def test_values_transposed_correctly(self):
        df = pd.DataFrame({"sample": ["s1", "s2"], "x": [1, 3]})
        result = _run(df)
        x_row = result[result["column"] == "x"].iloc[0]
        assert int(x_row["0"]) == 1
        assert int(x_row["1"]) == 3

    def test_round_trip_shape(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        once = _run(df)
        assert once.shape == (3, 4)  # 3 cols -> 3 rows; 3 rows + keycol -> 4 cols

    def test_single_row_dataframe(self):
        df = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        result = _run(df)
        assert len(result) == 3
        assert list(result.columns) == ["column", "0"]

    def test_single_column_dataframe(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = _run(df)
        assert len(result) == 1
        assert list(result["column"]) == ["x"]
