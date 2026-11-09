"""
Tests for dftk append: value parsing, adding a row, and --inplace logs.
"""

import argparse
import json
import multiprocessing

import pandas as pd
import pytest

import dftk.commands.append_cmd as append_cmd
from dftk.commands.append_cmd import AppendCommand, parse_value


def _args(*argv):
    parser = argparse.ArgumentParser()
    AppendCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(*argv):
    AppendCommand().execute(_args(*argv))


@pytest.fixture(autouse=True)
def no_stdin(monkeypatch):
    """No piped input unless a test gives a DATAFILE."""
    monkeypatch.setattr(append_cmd, "_has_stdin_data", lambda: False)


def _json(capsys, *argv):
    _run(*argv, "--format", "json")
    return json.loads(capsys.readouterr().out)


class TestParseValue:
    @pytest.mark.parametrize(
        "text, value",
        [
            ("42", 42),
            ("-3", -3),
            ("1.5", 1.5),
            ("1e-3", 0.001),
            (".5", 0.5),
            ("true", True),
            ("FALSE", False),
            ('"007"', "007"),
            ("'a b'", "a b"),
            ("", None),
            ("007", 7),
            ("1_000", "1_000"),
            ("nan", "nan"),
            ("S12", "S12"),
        ],
    )
    def test_value(self, text, value):
        assert parse_value(text) == value
        assert type(parse_value(text)) is type(value)


class TestNewTable:
    def test_one_row_tsv(self, capsys):
        _run("-c", "sample=S12", "-c", "reads = 48213")
        assert capsys.readouterr().out.splitlines() == [
            "sample\treads",
            "S12\t48213",
        ]

    def test_json_types(self, capsys):
        records = _json(
            capsys, "-c", "n=3", "-c", "ok=true", "-c", 'id="007"', "-c", "note="
        )
        assert records == [{"n": 3, "ok": True, "id": "007", "note": None}]

    def test_timestamp(self, capsys):
        records = _json(capsys, "-c", "step=a", "--timestamp")
        stamp = pd.Timestamp(records[0]["timestamp"])
        assert abs(pd.Timestamp.now() - stamp) < pd.Timedelta(minutes=1)

    def test_timestamp_column_name(self, capsys):
        assert "done_at" in _json(capsys, "--timestamp", "done_at")[0]

    @pytest.mark.parametrize("item", ["novalue", "=5"])
    def test_bad_item(self, item):
        with pytest.raises(ValueError, match="COL=VALUE"):
            _run("-c", item)

    def test_nothing_to_append(self):
        with pytest.raises(ValueError, match="nothing to append"):
            _run()


class TestAddToTable:
    @pytest.fixture
    def data(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text("name\tn\tx\na\t1\t0.5\nb\t2\t1.5\n")
        return str(path)

    def test_missing_columns_empty_new_column_last(self, data, capsys):
        records = _json(capsys, data, "-c", "name=c", "-c", "extra=yes")
        assert records[-1] == {"name": "c", "n": None, "x": None, "extra": "yes"}
        assert records[0]["extra"] is None

    def test_integer_column_stays_integer(self, data, capsys):
        records = _json(capsys, data, "-c", "name=c")
        assert [r["n"] for r in records] == [1, 2, None]

    def test_number_in_text_column_is_text(self, data, capsys):
        records = _json(capsys, data, "-c", "name=8")
        assert records[-1]["name"] == "8"

    def test_text_in_number_column_makes_it_text(self, data, capsys, caplog):
        records = _json(capsys, data, "-c", "n=many")
        assert [r["n"] for r in records] == ["1", "2", "many"]
        assert "makes it a text column" in caplog.text

    def test_fill(self, data, capsys):
        records = _json(capsys, data, "-c", "name=c", "-c", "k=1", "--fill", "0")
        assert records[-1]["x"] == 0
        assert records[0]["k"] == 0

    def test_strict(self, data):
        with pytest.raises(ValueError, match=r"--strict: \['nmae'\]"):
            _run(data, "-c", "nmae=c", "--strict")


class TestInplaceTsv:
    def test_creates_file(self, tmp_path):
        log = str(tmp_path / "runs.tsv")
        _run(log, "--inplace", "-c", "step=align", "-c", "n=5")
        assert open(log).read() == "step\tn\nalign\t5\n"

    def test_known_columns_add_one_line(self, tmp_path):
        log = tmp_path / "runs.tsv"
        log.write_text("step\tn\tnote\nalign\t5.0\tx\n")
        _run(str(log), "--inplace", "-c", "n=7", "-c", "step=call")
        assert log.read_text() == "step\tn\tnote\nalign\t5.0\tx\ncall\t7\t\n"

    def test_new_column_keeps_old_cells_as_written(self, tmp_path):
        log = tmp_path / "ids.tsv"
        log.write_text("id\tscore\n007\t1.50\n")
        _run(str(log), "--inplace", "-c", "id=8", "-c", "flag=true")
        assert log.read_text().splitlines() == [
            "id\tscore\tflag",
            "007\t1.50\t",
            "8\t\tTrue",
        ]

    def test_file_without_final_newline(self, tmp_path):
        log = tmp_path / "runs.tsv"
        log.write_text("a\tb\n1\t2")
        _run(str(log), "--inplace", "-c", "a=3")
        assert log.read_text() == "a\tb\n1\t2\n3\t\n"

    def test_strict_needs_the_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            _run(str(tmp_path / "typo.tsv"), "--inplace", "-c", "a=1", "--strict")

    def test_strict_refuses_new_column(self, tmp_path):
        log = tmp_path / "runs.tsv"
        log.write_text("step\tstatus\n")
        with pytest.raises(ValueError, match="stauts"):
            _run(str(log), "--inplace", "-c", "stauts=ok", "--strict")
        assert log.read_text() == "step\tstatus\n"

    @pytest.mark.parametrize(
        "extra, option",
        [
            (["-o", "x.tsv"], "--output"),
            (["--prequery", "a > 1"], "--prequery"),
            (["--select", "a"], "--select"),
        ],
    )
    def test_output_and_read_options_refused(self, tmp_path, extra, option):
        log = str(tmp_path / "runs.tsv")
        with pytest.raises(ValueError, match=f"can't be combined with {option}"):
            _run(log, "--inplace", "-c", "a=1", *extra)

    def test_needs_a_file(self):
        with pytest.raises(ValueError, match="needs a FILE"):
            _run("--inplace", "-c", "a=1")

    @pytest.mark.parametrize("ext", [".json", ".md", ".tex"])
    def test_document_formats_refused(self, tmp_path, ext):
        with pytest.raises(ValueError, match="works on TSV, .jsonl and .parquet"):
            _run(str(tmp_path / f"log{ext}"), "--inplace", "-c", "a=1")

    def test_keeps_file_mode(self, tmp_path):
        log = tmp_path / "runs.tsv"
        log.write_text("a\n1\n")
        log.chmod(0o640)
        _run(str(log), "--inplace", "-c", "b=2")  # new column: rewrite
        assert log.stat().st_mode & 0o777 == 0o640
        assert [p.name for p in tmp_path.iterdir()] == ["runs.tsv"]


class TestInplaceJsonl:
    def test_records_need_not_match(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        _run(str(log), "--inplace", "-c", "step=align", "-c", "reads=48213")
        _run(str(log), "--inplace", "-c", "step=call", "-c", "ok=true")
        assert [json.loads(line) for line in log.read_text().splitlines()] == [
            {"step": "align", "reads": 48213},
            {"step": "call", "ok": True},
        ]

    def test_strict_uses_keys_of_all_records(self, tmp_path):
        log = tmp_path / "runs.jsonl"
        log.write_text('{"a": 1}\n{"b": 2}\n')
        _run(str(log), "--inplace", "-c", "b=3", "--strict")
        with pytest.raises(ValueError, match=r"\['c'\]"):
            _run(str(log), "--inplace", "-c", "c=3", "--strict")


class TestInplaceParquet:
    def test_rows_and_types(self, tmp_path):
        log = str(tmp_path / "runs.parquet")
        _run(log, "--inplace", "-c", "step=a", "-c", "n=1")
        _run(log, "--inplace", "-c", "step=b", "-c", "x=2.5")
        df = pd.read_parquet(log)
        assert df["step"].tolist() == ["a", "b"]
        assert df["n"].tolist()[0] == 1 and pd.isna(df["n"].iloc[1])
        assert pd.api.types.is_integer_dtype(df["n"])

    def test_keeps_annotations(self, tmp_path):
        from dftk.common.io import _read_parquet_meta, _write_parquet

        log = str(tmp_path / "runs.parquet")
        _write_parquet(pd.DataFrame({"a": [1]}), log, meta={"source": "test"})
        _run(log, "--inplace", "-c", "a=2")
        assert _read_parquet_meta(log)["source"] == "test"


def _append_job(log, i):
    _run(log, "--inplace", "-c", f"job={i}", *(["-c", f"c{i}=x"] if i % 3 == 0 else []))


class TestParallelWriters:
    @pytest.mark.parametrize("name", ["runs.tsv", "runs.jsonl", "runs.parquet"])
    def test_no_row_is_lost(self, tmp_path, name):
        log = str(tmp_path / name)
        ctx = multiprocessing.get_context("spawn")
        jobs = [ctx.Process(target=_append_job, args=(log, i)) for i in range(8)]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()
            assert job.exitcode == 0
        if name.endswith(".parquet"):
            done = pd.read_parquet(log)["job"].tolist()
        elif name.endswith(".jsonl"):
            done = [json.loads(line)["job"] for line in open(log)]
        else:
            done = pd.read_csv(log, sep="\t")["job"].tolist()
        assert sorted(done) == list(range(8))
