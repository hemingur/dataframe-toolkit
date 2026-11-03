"""Tests for dftk.commands.print_cmd (dftk print)."""

import pandas as pd

from dftk.commands.print_cmd import PrintCommand
from tests.conftest import make_args


def _print(capsys, **kwargs):
    PrintCommand().execute(make_args(**kwargs))
    return capsys.readouterr().out


class TestPrint:
    def test_parquet_to_tsv(self, tmp_path, capsys):
        path = tmp_path / "x.parquet"
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_parquet(path)
        assert _print(capsys, DATAFILE=str(path)) == "a\tb\n1\tx\n2\ty\n"

    def test_names_columns_of_a_headerless_file(self, tmp_path, capsys):
        path = tmp_path / "data.txt"
        path.write_text("chr1\t100\nchr2\t50\n")
        out = _print(
            capsys, DATAFILE=str(path), noheader=True, move=["V1:chr", "V2:pos"]
        )
        assert out == "chr\tpos\nchr1\t100\nchr2\t50\n"

    def test_removeheader(self, tmp_path, capsys):
        path = tmp_path / "data.tsv"
        path.write_text("a\tb\n1\t2\n")
        assert _print(capsys, DATAFILE=str(path), removeheader=True) == "1\t2\n"

    def test_select(self, tmp_path, capsys):
        path = tmp_path / "data.tsv"
        path.write_text("a\tb\n1\t2\n")
        assert _print(capsys, DATAFILE=str(path), select=["b"]) == "b\n2\n"

    def test_one_column_missing_value_is_an_empty_line(self, tmp_path, capsys):
        # print used to write "" here, unlike every other command
        path = tmp_path / "x.parquet"
        pd.DataFrame({"y": [1.0, None]}).to_parquet(path)
        assert _print(capsys, DATAFILE=str(path)) == "y\n1.0\n\n"
