"""
Tests for dftk.common.io.read.
"""

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
