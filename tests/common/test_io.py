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
            "pandas.DataFrame.to_csv", lambda self, *a, **k: seen.update(df=self)
        )
        io.printdf(pd.DataFrame({"a": [1, 2]}), make_args(cast=[f"a:{spec}"]))
        assert str(seen["df"]["a"].dtype) == dtype

    def test_non_whole_float_to_int64_warns(self, capsys, caplog):
        df = pd.DataFrame({"k": ["a", "b"], "v": [2.5, None]})
        out = _printdf(df, capsys, cast=["v:Int64"])
        assert "--cast: failed for 'v:Int64'" in caplog.text
        assert out.splitlines()[1] == "a\t2.5"
