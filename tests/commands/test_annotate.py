"""Tests for stattools.commands.annotate_cmd and parquet metadata I/O."""

import os
import tempfile

import pandas as pd
import pytest

from stattools.commands.annotate_cmd import _read_meta, _write_meta
from stattools.common.io import _read_parquet_meta, _write_parquet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def parquet_file(tmp_path):
    """A parquet file with two columns and no custom metadata."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    path = str(tmp_path / "data.parquet")
    _write_parquet(df, path)
    return path


@pytest.fixture
def annotated_file(tmp_path):
    """A parquet file with pre-existing custom metadata."""
    df = pd.DataFrame({"x": [1, 2, 3]})
    path = str(tmp_path / "annotated.parquet")
    _write_parquet(df, path, meta={"genome": "hg38", "source": "test"})
    return path


# ---------------------------------------------------------------------------
# _write_parquet / _read_parquet_meta (io helpers)
# ---------------------------------------------------------------------------


class TestWriteParquet:

    def test_round_trip_data(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = str(tmp_path / "out.parquet")
        _write_parquet(df, path)
        result = pd.read_parquet(path)
        assert list(result.columns) == ["a", "b"]
        assert list(result["a"]) == [1, 2]

    def test_embeds_meta(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = str(tmp_path / "out.parquet")
        _write_parquet(df, path, meta={"genome": "hg38", "source": "gwas"})
        meta = _read_parquet_meta(path)
        assert meta["genome"] == "hg38"
        assert meta["source"] == "gwas"

    def test_no_meta_returns_empty(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = str(tmp_path / "out.parquet")
        _write_parquet(df, path)
        meta = _read_parquet_meta(path)
        assert meta == {}

    def test_pandas_key_excluded(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        path = str(tmp_path / "out.parquet")
        _write_parquet(df, path)
        meta = _read_parquet_meta(path)
        assert "pandas" not in meta

    def test_meta_overrides_attrs(self, tmp_path):
        """Explicit meta= takes precedence over df.attrs."""
        df = pd.DataFrame({"a": [1]})
        df.attrs["_parquet_meta"] = {"genome": "hg19", "source": "old"}
        path = str(tmp_path / "out.parquet")
        _write_parquet(df, path, meta={"genome": "hg38"})
        meta = _read_parquet_meta(path)
        assert meta["genome"] == "hg38"   # overridden
        assert meta["source"] == "old"    # carried from attrs


class TestMetaPropagation:

    def test_attrs_carried_through_write(self, tmp_path):
        """Metadata loaded into df.attrs is re-embedded on the next write."""
        df = pd.DataFrame({"a": [1, 2]})
        p1 = str(tmp_path / "step1.parquet")
        p2 = str(tmp_path / "step2.parquet")

        _write_parquet(df, p1, meta={"genome": "hg38", "source": "original"})

        df2 = pd.read_parquet(p1)
        df2.attrs["_parquet_meta"] = _read_parquet_meta(p1)
        df2["b"] = df2["a"] * 2

        _write_parquet(df2, p2, meta={"step": "doubled"})
        meta2 = _read_parquet_meta(p2)

        assert meta2["genome"] == "hg38"        # propagated
        assert meta2["source"] == "original"    # propagated
        assert meta2["step"] == "doubled"       # new


# ---------------------------------------------------------------------------
# _read_meta / _write_meta (annotate_cmd helpers)
# ---------------------------------------------------------------------------


class TestReadMeta:

    def test_reads_custom_keys(self, annotated_file):
        meta = _read_meta(annotated_file)
        assert meta["genome"] == "hg38"
        assert meta["source"] == "test"

    def test_empty_when_no_annotations(self, parquet_file):
        assert _read_meta(parquet_file) == {}

    def test_excludes_pandas_key(self, annotated_file):
        meta = _read_meta(annotated_file)
        assert "pandas" not in meta


class TestWriteMeta:

    def test_sets_keys(self, parquet_file):
        _write_meta(parquet_file, {"genome": "mm10"})
        assert _read_meta(parquet_file)["genome"] == "mm10"

    def test_replaces_existing(self, annotated_file):
        _write_meta(annotated_file, {"genome": "mm10"})
        meta = _read_meta(annotated_file)
        assert meta["genome"] == "mm10"
        assert "source" not in meta   # replaced entirely

    def test_preserves_pandas_key(self, annotated_file):
        """_write_meta must not corrupt the pandas schema metadata."""
        _write_meta(annotated_file, {"x": "y"})
        df = pd.read_parquet(annotated_file)
        assert list(df.columns) == ["x"]   # data intact

    def test_clear_by_empty_dict(self, annotated_file):
        _write_meta(annotated_file, {})
        assert _read_meta(annotated_file) == {}


# ---------------------------------------------------------------------------
# AnnotateCommand.execute (via CLI helpers)
# ---------------------------------------------------------------------------


from stattools.commands.annotate_cmd import AnnotateCommand
import argparse


def _make_args(**kwargs):
    defaults = dict(
        PARQUET=None,
        set_meta=[],
        get_key=None,
        delete_keys=[],
        clear=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestAnnotateExecute:

    def test_list_prints_sorted_tsv(self, annotated_file, capsys):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file))
        out = capsys.readouterr().out
        lines = out.strip().splitlines()
        assert lines[0] == "genome\thg38"
        assert lines[1] == "source\ttest"

    def test_list_empty_produces_no_output(self, parquet_file, capsys):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=parquet_file))
        assert capsys.readouterr().out == ""

    def test_set_adds_key(self, parquet_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=parquet_file, set_meta=["genome=hg38"]))
        assert _read_meta(parquet_file)["genome"] == "hg38"

    def test_set_multiple_keys(self, parquet_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=parquet_file,
                               set_meta=["genome=hg38", "source=test"]))
        meta = _read_meta(parquet_file)
        assert meta["genome"] == "hg38"
        assert meta["source"] == "test"

    def test_set_updates_existing(self, annotated_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file, set_meta=["genome=mm10"]))
        assert _read_meta(annotated_file)["genome"] == "mm10"

    def test_delete_removes_key(self, annotated_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file, delete_keys=["source"]))
        meta = _read_meta(annotated_file)
        assert "source" not in meta
        assert meta["genome"] == "hg38"   # other key intact

    def test_delete_missing_key_noop(self, annotated_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file, delete_keys=["nonexistent"]))
        meta = _read_meta(annotated_file)
        assert meta == {"genome": "hg38", "source": "test"}

    def test_clear_removes_all(self, annotated_file):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file, clear=True))
        assert _read_meta(annotated_file) == {}

    def test_get_prints_value(self, annotated_file, capsys):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=annotated_file, get_key="genome"))
        assert capsys.readouterr().out.strip() == "hg38"

    def test_get_missing_key_exits(self, annotated_file):
        cmd = AnnotateCommand()
        with pytest.raises(SystemExit):
            cmd.execute(_make_args(PARQUET=annotated_file, get_key="missing"))

    def test_set_then_get(self, parquet_file, capsys):
        cmd = AnnotateCommand()
        cmd.execute(_make_args(PARQUET=parquet_file,
                               set_meta=["step=normalized"],
                               get_key="step"))
        assert capsys.readouterr().out.strip() == "normalized"

    def test_non_parquet_exits(self, tmp_path):
        tsv = str(tmp_path / "data.tsv")
        open(tsv, "w").close()
        cmd = AnnotateCommand()
        with pytest.raises(SystemExit):
            cmd.execute(_make_args(PARQUET=tsv))
