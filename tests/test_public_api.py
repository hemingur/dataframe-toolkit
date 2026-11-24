"""
The interfaces other packages import from dftk.

A companion toolkit (mltk) builds its commands on dftk: it subclasses
BaseCommand, reads and writes with io.read / io.printdf, checks columns with
check_cols and shows DFTK_TMPDIR in its help.  Its commands then get DATAFILE
handling, the -o parquet pipe and the shared output options for free, and pipe
into dftk both ways.  These tests fail when one of those interfaces changes;
see "Interfaces other packages use" in CLAUDE.md before changing them.
"""

import argparse
import inspect
import io as stdio
import os
import sys

import pandas as pd
import pytest

from dftk.commands.base import BaseCommand
from dftk.common import io as io_module
from dftk.common.io import DFTK_TMPDIR, check_cols, io


class _Double(BaseCommand):
    """A command written the way an outside package writes one."""

    name = "double"
    help = "Double a column."

    def add_arguments(self, parser):
        self.add_io_arguments(parser)
        parser.add_argument("-c", "--col", required=True)

    def execute(self, args):
        df = io.read(args)
        check_cols(df, [args.col], "-c/--col")
        check_cols(df, None, "nothing")  # None is a no-op
        df[f"{args.col}2"] = df[args.col] * 2
        io.printdf(df, args)


def _parse(argv):
    parser = argparse.ArgumentParser()
    cmd = _Double()
    cmd.add_arguments(parser)
    return cmd, parser.parse_args(argv)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "t.tsv"
    path.write_text("a\tx\n1\t0.123\n2\t0.456\n")
    return path


class TestSignatures:
    def test_base_command(self):
        abstract = BaseCommand.__abstractmethods__
        assert abstract == {"name", "help", "add_arguments", "execute"}
        assert list(inspect.signature(BaseCommand.add_io_arguments).parameters) == [
            "self",
            "parser",
        ]

    def test_io_read_and_printdf(self):
        read = inspect.signature(io.read).parameters
        assert list(read)[0] == "args"
        assert all(
            p.default is not inspect.Parameter.empty for p in list(read.values())[1:]
        )
        assert list(inspect.signature(io.printdf).parameters) == ["df", "args"]

    def test_check_cols(self):
        assert list(inspect.signature(check_cols).parameters)[:3] == [
            "df",
            "cols",
            "context",
        ]

    def test_tmpdir(self):
        assert isinstance(DFTK_TMPDIR, str) and DFTK_TMPDIR


class TestOutsideCommand:
    def test_reads_a_file_and_writes_tsv(self, table, capsys):
        cmd, args = _parse([str(table), "-c", "a"])
        cmd.execute(args)
        out = capsys.readouterr().out.splitlines()
        assert out == ["a\tx\ta2", "1\t0.123\t2", "2\t0.456\t4"]

    @pytest.mark.parametrize(
        "options, expected",
        [
            (["--select", "a2"], ["a2", "2", "4"]),
            (["--round", "x:1", "--select", "x"], ["x", "0.1", "0.5"]),
            (["--postquery", "a > 1", "--select", "a"], ["a", "2"]),
            (["--format", "jsonl", "--select", "a2"], ['{"a2": 2}', '{"a2": 4}']),
        ],
    )
    def test_shared_output_options(self, table, capsys, options, expected):
        cmd, args = _parse([str(table), "-c", "a", *options])
        cmd.execute(args)
        assert capsys.readouterr().out.splitlines() == expected

    def test_missing_column_is_a_value_error(self, table):
        cmd, args = _parse([str(table), "-c", "nope"])
        with pytest.raises(ValueError, match="-c/--col"):
            cmd.execute(args)

    def test_parquet_pipe_both_ways(self, table, capsys, monkeypatch, tmp_path):
        monkeypatch.setattr(io_module, "DFTK_TMPDIR", str(tmp_path / "pipe"))
        # -o alone: a temp parquet, its path on stdout
        cmd, args = _parse([str(table), "-c", "a", "-o"])
        cmd.execute(args)
        path = capsys.readouterr().out.strip()
        assert path.endswith(".parquet") and os.path.exists(path)

        # the next command, with no DATAFILE, reads that path from stdin
        monkeypatch.setattr(
            sys, "stdin", stdio.TextIOWrapper(stdio.BytesIO(f"{path}\n".encode()))
        )
        cmd, args = _parse(["-c", "a2", "--select", "a2", "a22"])
        cmd.execute(args)
        assert capsys.readouterr().out.splitlines() == ["a2\ta22", "2\t4", "4\t8"]


def test_fit_output_columns():
    """Outside fit commands copy these names so their output pipes into forest."""
    from dftk.commands.fit_cmd import FitCommand

    df = pd.DataFrame({"y": [1.0, 2.1, 2.9, 4.2, 5.0], "x": [1, 2, 3, 4, 5]})
    parser = argparse.ArgumentParser()
    cmd = FitCommand()
    cmd.add_arguments(parser)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io, "read", staticmethod(lambda args, columns=None: df))
        captured = {}
        mp.setattr(
            io,
            "printdf",
            staticmethod(lambda out, args: captured.setdefault("df", out)),
        )
        cmd.execute(parser.parse_args(["-f", "y ~ x"]))
    assert list(captured["df"].columns[:4]) == ["tag", "variable", "marker", "effect"]
