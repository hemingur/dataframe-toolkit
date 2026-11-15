"""
The plots read only the parquet columns they use (issue #39).  Each test
writes a parquet file with columns the plot doesn't need and checks the
columns pd.read_parquet was asked for.
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.chi_cmd import ChiCommand  # noqa: E402
from dftk.commands.hist_cmd import HistCommand  # noqa: E402
from dftk.commands.line_cmd import LineCommand  # noqa: E402
from dftk.commands.manhattan_cmd import ManhattanCommand  # noqa: E402
from dftk.commands.qq_cmd import QqCommand  # noqa: E402
from dftk.commands.scat_cmd import ScatCommand  # noqa: E402


@pytest.fixture
def parquet_reads(monkeypatch):
    calls = []
    orig = pd.read_parquet

    def _spy(path, *a, columns=None, **kw):
        calls.append(columns)
        return orig(path, *a, columns=columns, **kw)

    monkeypatch.setattr(pd, "read_parquet", _spy)
    return calls


@pytest.fixture
def table(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "CHR": np.repeat(["1", "2", "3"], n // 3),
            "BP": np.tile(np.arange(1, n // 3 + 1) * 1000, 3),
            "P": rng.uniform(size=n),
            "CHISQ": rng.chisquare(1, size=n),
            "k": 1,
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "study": np.tile(["s1", "s2"], n // 2),
            "unused": "x",
            "unused2": 0.0,
        }
    )
    path = tmp_path / "t.parquet"
    df.to_parquet(path)
    return str(path)


def _run(command, path, tmp_path, *argv):
    parser = argparse.ArgumentParser()
    command.add_arguments(parser)
    args = parser.parse_args([path, *argv, "-f", str(tmp_path / "out.png")])
    command.execute(args)


@pytest.mark.parametrize(
    "command, argv, expected",
    [
        (ScatCommand(), ["-x", "a", "-y", "b", "--colorcol", "c"], {"a", "b", "c"}),
        (LineCommand(), ["-x", "a", "-y", "b", "-g", "study"], {"a", "b", "study"}),
        (HistCommand(), ["-x", "a", "--subgraphcol", "study"], {"a", "study"}),
        (ChiCommand(), ["-x", "a", "-y", "b"], {"a", "b"}),
        (ManhattanCommand(), [], {"CHR", "BP", "P"}),
        (
            ManhattanCommand(),
            ["--chi2col", "CHISQ", "--chi2df", "k", "--miami", "study"],
            {"CHR", "BP", "CHISQ", "k", "study"},
        ),
        (QqCommand(), [], {"P"}),
        (QqCommand(), ["--mlogpcol", "a", "-g", "study"], {"a", "study"}),
    ],
)
def test_only_used_columns_read(
    command, argv, expected, table, tmp_path, parquet_reads
):
    if "--mlogpcol" in argv:  # needs -log10(p) >= 0
        pd.read_parquet(table).assign(a=lambda d: d["a"].abs()).to_parquet(table)
        parquet_reads.clear()
    _run(command, table, tmp_path, *argv)
    (cols,) = parquet_reads
    assert set(cols) == expected


def test_prequery_reads_everything(table, tmp_path, parquet_reads):
    _run(ScatCommand(), table, tmp_path, "-x", "a", "-y", "b", "--prequery", "c > 0")
    assert parquet_reads == [None]


def test_unresolved_gwas_columns_read_everything(table, tmp_path, parquet_reads):
    """Without a position column the read is whole, and the error names them all."""
    df = pd.read_parquet(table).drop(columns="BP")
    df.to_parquet(table)
    parquet_reads.clear()
    with pytest.raises(ValueError, match="'unused'"):
        _run(ManhattanCommand(), table, tmp_path)
    assert parquet_reads == [None]


def test_missing_named_column_error_unchanged(table, tmp_path):
    with pytest.raises(ValueError, match="nosuch"):
        _run(ScatCommand(), table, tmp_path, "-x", "a", "-y", "nosuch")
