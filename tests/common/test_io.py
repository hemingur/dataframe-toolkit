"""
Tests for dftk.common.io: read and printdf.
"""

import io as _io
from types import SimpleNamespace

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
