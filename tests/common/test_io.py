"""
Tests for dftk.common.io: read and printdf.
"""

import io as _io
import os
import threading
import time
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

    @staticmethod
    def _fifo(tmp_path, text):
        # A named pipe stands in for <(...): DuckDB's second read of it is empty
        path = tmp_path / "pipe"
        os.mkfifo(path)

        def write():
            with open(path, "w") as fh:
                fh.write(text)
            # Give any later reader an EOF so a regression fails, not hangs
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                try:
                    os.close(os.open(path, os.O_WRONLY | os.O_NONBLOCK))
                except OSError:
                    time.sleep(0.01)

        threading.Thread(target=write, daemon=True).start()
        return path

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
    def test_reads_named_pipe(self, tmp_path):
        path = self._fifo(tmp_path, "id\tn\na\t10\nb\t20\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert df["id"].tolist() == ["a", "b"]
        assert df["n"].tolist() == [10, 20]

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
    def test_named_pipe_temp_file_is_removed(self, monkeypatch, tmp_path):
        spool = tmp_path / "spool"
        monkeypatch.setattr("dftk.common.io.DFTK_TMPDIR", str(spool))
        path = self._fifo(tmp_path, "x\n1\n")
        io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert list(spool.iterdir()) == []

    def test_hash_in_header_and_values_is_data(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("id\tCount #\tname\n1\t10\tA #1\n2\t20\tB\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert list(df.columns) == ["id", "Count #", "name"]
        assert df["name"].tolist() == ["A #1", "B"]

    def test_comment_line_in_data_raises(self, tmp_path):
        # DuckDB's comment guessing used to cut "Count #" to "Count" here,
        # or drop the header altogether
        path = tmp_path / "data.tsv"
        path.write_text("id\tCount #\tname\n1\t10\tA #1\n# note\n2\t20\tB\n")
        with pytest.raises(duckdb.Error):
            io.read(make_args(DATAFILE=str(path), backend="duckdb"))

    def test_comment_lines_above_header_are_skipped(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("# made by x\n# date y\nid\tn\na\t10\nb\t20\n")
        df = io.read(make_args(DATAFILE=str(path), backend="duckdb"))
        assert list(df.columns) == ["id", "n"]
        assert df["n"].tolist() == [10, 20]


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

    def test_reads_empty_line_as_missing(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("y\n10\n\n30\n")
        df = io.read(make_args(DATAFILE=str(path)))
        assert df["y"].tolist()[::2] == [10, 30]
        assert pd.isna(df["y"].iloc[1])

    def test_reads_stdin(self, monkeypatch):
        monkeypatch.setattr(
            "sys.stdin", SimpleNamespace(buffer=_io.BytesIO(b"y\na\n\nb\n"))
        )
        df = io.read(make_args(DATAFILE="-"))
        assert len(df) == 3

    def test_multi_column_file_still_skips_empty_lines(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("a\tb\n1\t2\n\n3\t4\n")
        assert len(io.read(make_args(DATAFILE=str(path)))) == 2

    def test_round_trip(self, tmp_path):
        path = tmp_path / "out.tsv"
        df = pd.DataFrame({"y": [10.0, None, None, 30.0]})
        io.printdf(df, make_args(output=str(path)))
        back = io.read(make_args(DATAFILE=str(path)))
        assert back["y"].tolist() == pytest.approx(df["y"].tolist(), nan_ok=True)


class TestEmptyAndDuplicateHeaders:
    """The default reader names them like pandas' C reader (pyarrow keeps them)."""

    @pytest.mark.parametrize(
        "header, expected",
        [
            ("a\t\ta\tb\ta", ["a", "Unnamed: 1", "a.1", "b", "a.2"]),
            ("a\ta\ta.1", ["a", "a.2", "a.1"]),
            ("a\ta.1\ta", ["a", "a.1", "a.2"]),
        ],
    )
    def test_names_match_pandas(self, tmp_path, header, expected):
        path = tmp_path / "data.tsv"
        ncols = len(header.split("\t"))
        path.write_text(header + "\n" + "\t".join(["1"] * ncols) + "\n")
        df = io.read(make_args(DATAFILE=str(path)))
        assert list(df.columns) == expected
        c_reader = pd.read_csv(path, sep="\t", engine="c")
        assert list(c_reader.columns) == expected

    def test_stdin(self, monkeypatch):
        monkeypatch.setattr(
            "sys.stdin", SimpleNamespace(buffer=_io.BytesIO(b"x\tx\n1\t2\n"))
        )
        assert list(io.read(make_args(DATAFILE="-")).columns) == ["x", "x.1"]

    def test_noheader_unaffected(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("a\ta\n1\t2\n")
        df = io.read(make_args(DATAFILE=str(path), noheader=True))
        assert list(df.columns) == ["V1", "V2"]


class TestStdinDetection:
    """stdin (or a pipe) holds either TSV or the parquet path printed by -o."""

    BACKENDS = ["pandas", "duckdb"]

    @staticmethod
    def _stdin(monkeypatch, data: bytes):
        monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=_io.BytesIO(data)))

    @staticmethod
    def _parquet(path, df=None):
        from dftk.common.io import _write_parquet

        df = pd.DataFrame({"id": ["a", "b"], "n": [1, 2]}) if df is None else df
        _write_parquet(df, str(path))
        return path

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("datafile", [None, "-"])
    def test_parquet_path_on_stdin(self, monkeypatch, tmp_path, backend, datafile):
        path = self._parquet(tmp_path / "x.parquet")
        self._stdin(monkeypatch, f"{path}\n".encode())
        df = io.read(make_args(DATAFILE=datafile, backend=backend))
        assert df.to_dict("list") == {"id": ["a", "b"], "n": [1, 2]}

    @pytest.mark.parametrize("ending", ["", "\r\n", "  \n\n"])
    def test_path_line_endings(self, monkeypatch, tmp_path, ending):
        path = self._parquet(tmp_path / "x.parquet")
        self._stdin(monkeypatch, f"{path}{ending}".encode())
        assert len(io.read(make_args(DATAFILE="-"))) == 2

    def test_temp_parquet_is_deleted(self, monkeypatch, tmp_path):
        monkeypatch.setattr("dftk.common.io.DFTK_TMPDIR", str(tmp_path))
        path = self._parquet(tmp_path / "tmpabc.parquet")
        self._stdin(monkeypatch, f"{path}\n".encode())
        io.read(make_args(DATAFILE=None))
        assert not path.exists()

    def test_named_parquet_is_kept(self, monkeypatch, tmp_path):
        monkeypatch.setattr("dftk.common.io.DFTK_TMPDIR", str(tmp_path / "tmp"))
        path = self._parquet(tmp_path / "keep.parquet")
        self._stdin(monkeypatch, f"{path}\n".encode())
        io.read(make_args(DATAFILE=None))
        assert path.exists()

    def test_metadata_is_carried(self, monkeypatch, tmp_path):
        from dftk.common.io import _write_parquet

        path = tmp_path / "x.parquet"
        _write_parquet(pd.DataFrame({"a": [1]}), str(path), meta={"step": "qc"})
        self._stdin(monkeypatch, f"{path}\n".encode())
        df = io.read(make_args(DATAFILE=None))
        assert df.attrs["_parquet_meta"]["step"] == "qc"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_tsv_on_stdin_is_unchanged(self, monkeypatch, backend):
        self._stdin(monkeypatch, b"id\tn\na\t1\nb\t2\n")
        df = io.read(make_args(DATAFILE=None, backend=backend))
        assert df.to_dict("list") == {"id": ["a", "b"], "n": [1, 2]}

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_header_only_tsv(self, monkeypatch, backend):
        self._stdin(monkeypatch, b"id\tn\n")
        df = io.read(make_args(DATAFILE="-", backend=backend))
        assert list(df.columns) == ["id", "n"]
        assert len(df) == 0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_path_followed_by_data_is_tsv(self, monkeypatch, tmp_path, backend):
        # A one-column TSV whose header happens to name an existing file
        path = self._parquet(tmp_path / "x.parquet")
        self._stdin(monkeypatch, f"{path}\nv1\nv2\n".encode())
        df = io.read(make_args(DATAFILE="-", backend=backend))
        assert list(df.columns) == [str(path)]
        assert df.iloc[:, 0].tolist() == ["v1", "v2"]

    def test_missing_parquet_path_is_tsv(self, monkeypatch, tmp_path):
        self._stdin(monkeypatch, f"{tmp_path / 'gone.parquet'}\n".encode())
        df = io.read(make_args(DATAFILE="-"))
        assert len(df) == 0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_first_line_longer_than_path_limit(self, monkeypatch, backend):
        from dftk.common.io import _MAX_PATH_LINE

        cols = [f"c{i:05d}" for i in range(_MAX_PATH_LINE // 6 + 10)]
        data = "\t".join(cols) + "\n" + "\t".join("1" for _ in cols) + "\n"
        assert len(data.split("\n")[0]) > _MAX_PATH_LINE
        self._stdin(monkeypatch, data.encode())
        df = io.read(make_args(DATAFILE="-", backend=backend))
        assert list(df.columns) == cols
        assert df.iloc[0].tolist() == [1] * len(cols)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_crlf_tsv(self, monkeypatch, backend):
        self._stdin(monkeypatch, b"id\tn\r\na\t1\r\nb\t2\r\n")
        df = io.read(make_args(DATAFILE="-", backend=backend))
        assert df["n"].tolist() == [1, 2]

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_slow_pipe_loses_nothing(self, monkeypatch, backend):
        # Rows written in pieces with pauses must all arrive, first line too
        rows = [f"r{i}\t{i}\n" for i in range(5000)]
        data = ("id\tn\n" + "".join(rows)).encode()
        read_fd, write_fd = os.pipe()

        def write():
            with os.fdopen(write_fd, "wb") as fh:
                for start in range(0, len(data), 7000):
                    fh.write(data[start : start + 7000])
                    fh.flush()
                    time.sleep(0.005)

        threading.Thread(target=write, daemon=True).start()
        with os.fdopen(read_fd, "rb") as reader:
            monkeypatch.setattr("sys.stdin", SimpleNamespace(buffer=reader))
            df = io.read(make_args(DATAFILE="-", backend=backend))
        assert len(df) == 5000
        assert df["id"].iloc[0] == "r0"
        assert df["n"].iloc[-1] == 4999

    def test_ellipsis_still_requires_a_parquet_path(self, monkeypatch):
        monkeypatch.setattr("sys.stdin", _io.StringIO("id\tn\n"))
        with pytest.raises(ValueError, match="expects a .parquet path"):
            io.read(make_args(DATAFILE="..."))

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
    @pytest.mark.parametrize("backend", BACKENDS)
    def test_named_pipe_with_parquet_path(self, tmp_path, backend):
        # <(dftk ... -o): the pipe holds a path, not data
        path = self._parquet(tmp_path / "x.parquet")
        pipe = TestReadDuckdbBackend._fifo(tmp_path, f"{path}\n")
        df = io.read(make_args(DATAFILE=str(pipe), backend=backend))
        assert df.to_dict("list") == {"id": ["a", "b"], "n": [1, 2]}

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
    def test_named_pipe_one_column_keeps_empty_lines(self, tmp_path):
        # The pandas reader used to read a pipe twice for one-column files
        pipe = TestReadDuckdbBackend._fifo(tmp_path, "y\na\n\nb\n")
        df = io.read(make_args(DATAFILE=str(pipe)))
        assert len(df) == 3

    def test_directory_is_an_error_naming_the_shell(self, tmp_path):
        with pytest.raises(ValueError, match=r"directory.*\.\.\..*stdin"):
            io.read(make_args(DATAFILE=str(tmp_path)))
