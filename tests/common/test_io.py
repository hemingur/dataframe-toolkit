"""
Tests for dftk.common.io: read and printdf.
"""

import io as _io
from types import SimpleNamespace

import pandas as pd
import pytest

from dftk.common.io import io
from tests.conftest import make_args

duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")


class TestReadDuckdbBackend:
    def test_reads_named_tsv(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("id\tx\na\t1.5\nb\t2.5\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert list(df.columns) == ["id", "x"]
        assert df["id"].tolist() == ["a", "b"]
        assert df["x"].tolist() == [1.5, 2.5]

    def test_custom_delimiter(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("id,x\na,1\nb,2\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb", delimiter=","))
        assert list(df.columns) == ["id", "x"]
        assert df["x"].tolist() == [1, 2]

    def test_noheader_names_columns_v1_v2(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("a\t1\nb\t2\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb", noheader=True))
        assert list(df.columns) == ["V1", "V2"]
        assert df["V1"].tolist() == ["a", "b"]

    def test_integer_column_with_blank_stays_integer(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("id\tn\n007\t10\n008\t\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert str(df["n"].dtype) == "Int64"
        assert df["n"].iloc[0] == 10
        assert df["id"].tolist() == ["007", "008"]

    def test_reads_stdin(self, monkeypatch):
        stdin = SimpleNamespace(buffer=_io.BytesIO(b"id\tn\n007\t10\n008\t\n"))
        monkeypatch.setattr("sys.stdin", stdin)
        df = io.read(make_args(DATAFILE="-", backend="duckdb"))
        assert str(df["n"].dtype) == "Int64"
        assert df["id"].tolist() == ["007", "008"]

    def test_stdin_temp_file_is_removed(self, monkeypatch, tmp_path):
        monkeypatch.setattr("dftk.common.io.DFTK_TMPDIR", str(tmp_path))
        monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=_io.BytesIO(b"x\n1\n")))
        io.read(make_args(DATAFILE="-", backend="duckdb"))
        assert list(tmp_path.iterdir()) == []


def _printdf(df, capsys, **overrides):
    """Run io.printdf on *df* and return its TSV output."""
    io.printdf(df, make_args(**overrides))
    return capsys.readouterr().out


class TestCast:
    def test_int64_keeps_empty_cells(self, capsys):
        df = pd.DataFrame({"k": ["a", "b"], "pos": [248956422.0, None]})
        out = _printdf(df, capsys, cast=["pos:Int64"])
        assert out == "k\tpos\na\t248956422\nb\t\n"

    @pytest.mark.parametrize(
        "spec, dtype",
        [
            ("int", "int64"),
            ("Int", "Int64"),
            ("float", "float64"),
            ("object", "object"),
            ("category", "category"),
            ("datetime64", "datetime64[ns]"),
        ],
    )
    def test_type_names(self, spec, dtype, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            "dftk.common.io._write_tsv", lambda df, *a, **k: seen.update(df=df)
        )
        io.printdf(pd.DataFrame({"a": [1, 2]}), make_args(cast=[f"a:{spec}"]))
        assert str(seen["df"]["a"].dtype) == dtype

    def test_non_whole_float_to_int64_warns(self, capsys, caplog):
        df = pd.DataFrame({"k": ["a", "b"], "v": [2.5, None]})
        out = _printdf(df, capsys, cast=["v:Int64"])
        assert "--cast: failed for 'v:Int64'" in caplog.text
        assert out.splitlines()[1] == "a\t2.5"


class TestDigitsWarning:
    def test_warns_and_suggests_cast(self, capsys, caplog):
        df = pd.DataFrame({"pos": [248956422.0, None], "my start": [1000001.0, None]})
        out = _printdf(df, capsys, digits=4)
        assert "column(s) 'pos', 'my start'" in caplog.text
        assert "248956422 -> 2.49e+08" in caplog.text
        assert "--cast pos:Int64 'my start:Int64'" in caplog.text
        assert "--round 4" in caplog.text
        # the warning does not change the output
        assert out.splitlines()[1] == "2.49e+08\t1e+06"

    def test_suggested_cast_avoids_exponent_form(self, capsys, caplog):
        df = pd.DataFrame({"k": ["a", "b"], "pos": [248956422.0, None]})
        out = _printdf(df, capsys, digits=4, cast=["pos:Int64"])
        assert out.splitlines()[1] == "a\t248956422"
        assert "--digits" not in caplog.text

    @pytest.mark.parametrize(
        "values",
        [[9999.0, None], [123456789.5, 1.5], [float("inf"), 1.0]],
        ids=["fits", "not-whole", "inf"],
    )
    def test_no_warning(self, values, capsys, caplog):
        _printdf(pd.DataFrame({"k": ["a", "b"], "x": values}), capsys, digits=4)
        assert "--digits" not in caplog.text

    def test_lists_three_columns_then_count(self, capsys, caplog):
        df = pd.DataFrame({f"c{i}": [1e6, None] for i in range(5)})
        _printdf(df, capsys, digits=2)
        assert "'c0', 'c1', 'c2' and 2 more" in caplog.text
        assert "--cast c0:Int64 c1:Int64 c2:Int64 ..." in caplog.text

    def test_no_warning_for_parquet_output(self, tmp_path, capsys, caplog):
        df = pd.DataFrame({"pos": [248956422.0, None]})
        _printdf(df, capsys, digits=4, output=str(tmp_path / "out.parquet"))
        assert "--digits" not in caplog.text


class TestSingleColumnTsv:
    """One column: a missing value is an empty line, both written and read."""

    def test_writes_empty_line_for_missing_value(self, capsys):
        out = _printdf(pd.DataFrame({"y": [10.0, None, 30.0]}), capsys)
        assert out == "y\n10.0\n\n30.0\n"

    def test_writes_named_file(self, tmp_path):
        path = tmp_path / "out.tsv"
        io.printdf(pd.DataFrame({"y": ["a", ""]}), make_args(output=str(path)))
        assert path.read_text() == "y\na\n\n"

    def test_options_still_apply(self, capsys):
        df = pd.DataFrame({"y": [1 / 3, None]})
        assert _printdf(df, capsys, digits=2, removeheader=True) == "0.33\n\n"

    def test_quote_characters_unchanged(self, capsys):
        assert _printdf(pd.DataFrame({"v": ['x"y', None]}), capsys) == 'v\nx"y\n\n'

    def test_padding_name_does_not_clash(self, capsys):
        assert _printdf(pd.DataFrame({"_pad": [1.0, None]}), capsys) == "_pad\n1.0\n\n"
