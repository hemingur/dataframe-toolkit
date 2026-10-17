"""Tests for dftk.commands.cleannames_cmd."""

import json

import pandas as pd
import pyarrow.parquet as pq
import pytest

from dftk.commands.cleannames_cmd import CleannamesCommand, clean_name, clean_names
from tests.conftest import make_args

# ---------------------------------------------------------------------------
# clean_name() — one name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Sample ID", "sample_id"),
        (" Age (years) ", "age_years"),
        ("GC %", "gc_pct"),
        ("GC%", "gc_pct"),
        ("Count #", "count_n"),
        ("#reads", "n_reads"),
        ("1st dose", "x1st_dose"),
        ("1960", "x1960"),
        ("Hæð (cm)", "haed_cm"),
        ("Þyngd", "thyngd"),
        ("Ölvun", "olvun"),
        ("conc. (µg/ml)", "conc_ug_ml"),
        ("Straße", "strasse"),
        ("sampleID", "sampleid"),
        ("HbA1c", "hba1c"),
        ("a--b__c", "a_b_c"),
        ("already_clean", "already_clean"),
        ("---", ""),
        ("", ""),
    ],
)
def test_clean_name(name, expected):
    assert clean_name(name) == expected


# ---------------------------------------------------------------------------
# clean_names() — a header row
# ---------------------------------------------------------------------------


class TestCleanNames:
    def test_nothing_left_becomes_col_and_position(self):
        assert clean_names(["a", "---", "()"]) == ["a", "col2", "col3"]

    def test_clashes_get_suffix(self):
        assert clean_names(["GC %", "gc_pct", "GC%"]) == [
            "gc_pct",
            "gc_pct_2",
            "gc_pct_3",
        ]

    def test_suffix_skips_existing_name(self):
        # "a_2" is a header of its own, so the second "A" becomes a_3
        assert clean_names(["a", "A", "a_2"]) == ["a", "a_3", "a_2"]

    def test_reader_duplicates_stay_numbered(self):
        # the reader turns a second "Count #" into "Count #.1" (DuckDB: "Count #_1")
        assert clean_names(["Count #", "Count #.1"]) == ["count_n", "count_n_1"]
        assert clean_names(["Count #", "Count #_1"]) == ["count_n", "count_n_1"]

    def test_names_are_unique(self):
        names = ["x", "X", "x ", "x_2", "", "", "1", "x1"]
        assert len(set(clean_names(names))) == len(names)


# ---------------------------------------------------------------------------
# CleannamesCommand
# ---------------------------------------------------------------------------


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "data.tsv"
    path.write_text(text)
    return str(path)


def _args(**kwargs):
    defaults = dict(mapping=None)
    defaults.update(kwargs)
    return make_args(**defaults)


class TestCleannamesCommand:
    def test_renames_columns_keeps_data(self, tmp_path, capsys):
        path = _write(tmp_path, "Sample ID\tGC %\nS1\t41.2\n")
        CleannamesCommand().execute(_args(DATAFILE=path))
        assert capsys.readouterr().out == "sample_id\tgc_pct\nS1\t41.2\n"

    def test_duplicate_and_empty_headers(self, tmp_path, capsys):
        path = _write(tmp_path, "Count #\t\tCount #\n1\t2\t3\n")
        CleannamesCommand().execute(_args(DATAFILE=path))
        header = capsys.readouterr().out.splitlines()[0]
        assert header == "count_n\tunnamed_1\tcount_n_1"

    def test_original_names_in_parquet_metadata(self, tmp_path, capsys):
        path = _write(tmp_path, "Sample ID\tage\nS1\t3\n")
        out = str(tmp_path / "out.parquet")
        CleannamesCommand().execute(_args(DATAFILE=path, output=out))
        meta = pq.read_schema(out).metadata
        assert json.loads(meta[b"original_names"]) == {"sample_id": "Sample ID"}

    def test_cleaning_again_keeps_first_originals(self, tmp_path, capsys):
        first = str(tmp_path / "first.parquet")
        df = pd.DataFrame({"Sample ID": ["S1"], "B": [1]})
        df.to_csv(tmp_path / "data.tsv", sep="\t", index=False)
        CleannamesCommand().execute(
            _args(DATAFILE=str(tmp_path / "data.tsv"), output=first)
        )
        second = str(tmp_path / "second.parquet")
        CleannamesCommand().execute(_args(DATAFILE=first, output=second))
        meta = json.loads(pq.read_schema(second).metadata[b"original_names"])
        assert meta == {"sample_id": "Sample ID", "b": "B"}

    def test_mapping_file(self, tmp_path, capsys):
        path = _write(tmp_path, "Sample ID\tage\nS1\t3\n")
        mapping = tmp_path / "names.tsv"
        CleannamesCommand().execute(_args(DATAFILE=path, mapping=str(mapping)))
        assert (
            mapping.read_text() == "original\tcleaned\nSample ID\tsample_id\nage\tage\n"
        )

    def test_warns_about_sql_reserved_words(self, tmp_path, capsys, caplog):
        path = _write(tmp_path, "Group\tName\tOrder\n1\t2\t3\n")
        CleannamesCommand().execute(_args(DATAFILE=path))
        # name is a keyword too, but DuckDB accepts it unquoted
        assert "cleannames: group, order are SQL reserved words" in caplog.text

    def test_no_warning_without_keywords(self, tmp_path, capsys, caplog):
        path = _write(tmp_path, "Sample ID\n1\n")
        CleannamesCommand().execute(_args(DATAFILE=path))
        assert "reserved" not in caplog.text

    def test_reserved_word_check_matches_duckdb(self):
        duckdb = pytest.importorskip("duckdb")
        from dftk.commands.cleannames_cmd import _sql_keywords

        df = pd.DataFrame({c: [1] for c in ["group", "left", "name", "position"]})
        for col in df.columns:
            fails = col in _sql_keywords([col])
            try:
                duckdb.sql(f"SELECT {col} FROM df").fetchall()
                assert not fails, col
            except duckdb.ParserException:
                assert fails, col
