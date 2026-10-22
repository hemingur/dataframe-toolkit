"""Tests for dftk.commands.explode_cmd."""

import io as _io
import sys

import pandas as pd
import pytest

from dftk.commands.explode_cmd import ExplodeCommand, _explode
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(cols=["v"], sep=",", poscol=None, dropempty=False)
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run ExplodeCommand on *df* and return the parsed TSV output."""
    import dftk.commands.explode_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ExplodeCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t", dtype=str)


# ---------------------------------------------------------------------------
# _explode() — direct function tests
# ---------------------------------------------------------------------------


class TestExplodeSplitting:
    def test_one_row_per_item_other_columns_repeated(self):
        df = pd.DataFrame({"gene": ["BRCA1", "TP53"], "v": ["GO:1,GO:2", "GO:3"]})
        result = _explode(df, _make_args())
        assert result["gene"].tolist() == ["BRCA1", "BRCA1", "TP53"]
        assert result["v"].tolist() == ["GO:1", "GO:2", "GO:3"]

    def test_index_is_renumbered(self):
        df = pd.DataFrame({"v": ["a,b", "c"]}, index=[10, 20])
        result = _explode(df, _make_args())
        assert result.index.tolist() == [0, 1, 2]

    def test_spaces_around_items_are_stripped(self):
        df = pd.DataFrame({"v": ["GO:1, GO:2 ,  GO:3"]})
        assert _explode(df, _make_args())["v"].tolist() == ["GO:1", "GO:2", "GO:3"]

    def test_other_separator(self):
        df = pd.DataFrame({"v": ["a;b;c"]})
        assert _explode(df, _make_args(sep=";"))["v"].tolist() == ["a", "b", "c"]

    def test_separator_is_not_a_regex(self):
        df = pd.DataFrame({"v": ["a|b", "c.d"]})
        assert _explode(df, _make_args(sep="|"))["v"].tolist() == ["a", "b", "c.d"]

    def test_multi_character_separator(self):
        df = pd.DataFrame({"v": ["a::b", "c:d"]})
        assert _explode(df, _make_args(sep="::"))["v"].tolist() == ["a", "b", "c:d"]

    def test_cell_without_separator_is_one_row(self):
        df = pd.DataFrame({"v": ["a"]})
        assert _explode(df, _make_args())["v"].tolist() == ["a"]

    def test_input_frame_is_not_modified(self):
        df = pd.DataFrame({"v": ["a,b"]})
        _explode(df, _make_args())
        assert df["v"].tolist() == ["a,b"]


class TestExplodeParallel:
    def test_items_are_paired_by_position(self):
        df = pd.DataFrame({"id": [1, 2], "a": ["x,y", "z"], "b": ["p,q", "r"]})
        result = _explode(df, _make_args(cols=["a", "b"]))
        assert result["id"].tolist() == [1, 1, 2]
        assert result["a"].tolist() == ["x", "y", "z"]
        assert result["b"].tolist() == ["p", "q", "r"]

    def test_different_item_counts_raise_naming_the_rows(self):
        df = pd.DataFrame({"a": ["x,y", "z", "u"], "b": ["p,q", "r,s", "t"]})
        with pytest.raises(ValueError, match=r"row 2 \(a: 1, b: 2\)"):
            _explode(df, _make_args(cols=["a", "b"]))

    def test_many_bad_rows_are_summarised(self):
        df = pd.DataFrame({"a": ["x"] * 8, "b": ["p,q"] * 8})
        with pytest.raises(ValueError, match="and 3 more"):
            _explode(df, _make_args(cols=["a", "b"]))

    def test_missing_cell_counts_as_one_item(self):
        df = pd.DataFrame({"a": ["x", None], "b": ["p", "q"]})
        result = _explode(df, _make_args(cols=["a", "b"]))
        assert result["b"].tolist() == ["p", "q"]
        assert pd.isna(result["a"].iloc[1])

    def test_column_listed_twice_raises(self):
        df = pd.DataFrame({"a": ["x,y"]})
        with pytest.raises(ValueError, match="more than once"):
            _explode(df, _make_args(cols=["a", "a"]))


class TestExplodeEmpty:
    def test_missing_cell_stays_as_one_row(self):
        df = pd.DataFrame({"id": [1, 2], "v": ["a,b", None]})
        result = _explode(df, _make_args())
        assert result["id"].tolist() == [1, 1, 2]
        assert pd.isna(result["v"].iloc[2])

    def test_empty_item_becomes_missing(self):
        df = pd.DataFrame({"v": ["a,,b"]})
        result = _explode(df, _make_args())
        assert len(result) == 3
        assert pd.isna(result["v"].iloc[1])

    def test_dropempty_removes_empty_and_missing(self):
        df = pd.DataFrame({"id": [1, 2], "v": ["a,,b", None]})
        result = _explode(df, _make_args(dropempty=True))
        assert result["v"].tolist() == ["a", "b"]
        assert result.index.tolist() == [0, 1]

    def test_dropempty_keeps_row_if_any_split_column_has_a_value(self):
        df = pd.DataFrame({"a": ["x,"], "b": [","]})
        result = _explode(df, _make_args(cols=["a", "b"], dropempty=True))
        assert result["a"].tolist() == ["x"]

    def test_empty_separator_raises(self):
        df = pd.DataFrame({"v": ["a"]})
        with pytest.raises(ValueError, match="--sep"):
            _explode(df, _make_args(sep=""))


class TestExplodePoscol:
    def test_positions_restart_on_each_input_row(self):
        df = pd.DataFrame({"v": ["a,b,c", "d"]})
        result = _explode(df, _make_args(poscol="n"))
        assert result["n"].tolist() == [1, 2, 3, 1]

    def test_positions_kept_with_dropempty(self):
        df = pd.DataFrame({"v": ["a,,b"]})
        result = _explode(df, _make_args(poscol="n", dropempty=True))
        assert result["n"].tolist() == [1, 3]

    def test_existing_column_name_raises(self):
        df = pd.DataFrame({"v": ["a"], "n": [1]})
        with pytest.raises(ValueError, match="already a column"):
            _explode(df, _make_args(poscol="n"))


class TestExplodeTypes:
    def test_integers_become_int64(self):
        df = pd.DataFrame({"v": ["1,2", "3"]})
        result = _explode(df, _make_args())
        assert result["v"].dtype == "int64"
        assert result["v"].tolist() == [1, 2, 3]

    def test_integers_with_gaps_become_nullable(self):
        df = pd.DataFrame({"v": ["1,,2", None]})
        assert str(_explode(df, _make_args())["v"].dtype) == "Int64"

    def test_floats_become_float64(self):
        df = pd.DataFrame({"v": ["1.5,2e3", "-3"]})
        result = _explode(df, _make_args())
        assert result["v"].dtype == "float64"
        assert result["v"].tolist() == [1.5, 2000.0, -3.0]

    def test_mixed_values_stay_text(self):
        df = pd.DataFrame({"v": ["1,a"]})
        assert _explode(df, _make_args())["v"].tolist() == ["1", "a"]

    def test_readasobject_column_stays_text(self):
        df = pd.DataFrame({"v": ["007,008"], "w": ["1,2"]})
        result = _explode(df, _make_args(cols=["v", "w"], readasobject=["v"]))
        assert result["v"].tolist() == ["007", "008"]
        assert result["w"].tolist() == [1, 2]

    def test_readasobject_without_names_keeps_all_text(self):
        df = pd.DataFrame({"v": ["007,008"]})
        result = _explode(df, _make_args(readasobject=[]))
        assert result["v"].tolist() == ["007", "008"]

    def test_numeric_column_is_kept_numeric(self):
        df = pd.DataFrame({"v": [1.5, None]})
        result = _explode(df, _make_args())
        assert result["v"].tolist()[0] == 1.5
        assert pd.isna(result["v"].iloc[1])


# ---------------------------------------------------------------------------
# ExplodeCommand.execute() — end to end
# ---------------------------------------------------------------------------


class TestExplodeExecute:
    def test_tsv_output(self):
        df = pd.DataFrame({"gene": ["BRCA1", "TP53"], "v": ["GO:1, GO:2", "GO:3"]})
        out = _run(df)
        assert out.to_dict("list") == {
            "gene": ["BRCA1", "BRCA1", "TP53"],
            "v": ["GO:1", "GO:2", "GO:3"],
        }

    def test_select_keeps_chosen_columns(self):
        df = pd.DataFrame({"gene": ["A"], "v": ["x,y"], "other": [1]})
        out = _run(df, select=["v", "gene"])
        assert list(out.columns) == ["v", "gene"]

    def test_missing_column_raises(self):
        df = pd.DataFrame({"v": ["a"]})
        with pytest.raises(ValueError, match="-c/--cols"):
            _run(df, cols=["nope"])
