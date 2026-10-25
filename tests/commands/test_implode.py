"""Tests for dftk.commands.implode_cmd."""

import io as _io
import sys

import pandas as pd
import pytest

from dftk.commands.explode_cmd import _explode
from dftk.commands.implode_cmd import ImplodeCommand, _implode
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(
        cols=["v"],
        indexcols=None,
        sep=",",
        poscol=None,
        unique=False,
        sortvalues=False,
    )
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run ImplodeCommand on *df* and return the parsed TSV output."""
    import dftk.commands.implode_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        ImplodeCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t", dtype=str)


def _values(s: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in s]


@pytest.fixture
def go_df():
    return pd.DataFrame(
        {
            "gene": ["BRCA1", "BRCA1", "TP53"],
            "v": ["GO:1", "GO:2", "GO:3"],
            "tissue": ["breast", "breast", "colon"],
        }
    )


# ---------------------------------------------------------------------------
# _implode() — direct function tests
# ---------------------------------------------------------------------------


class TestImplodeGrouping:
    def test_groups_by_all_other_columns(self, go_df):
        result = _implode(go_df, _make_args())
        assert result.to_dict("list") == {
            "gene": ["BRCA1", "TP53"],
            "v": ["GO:1,GO:2", "GO:3"],
            "tissue": ["breast", "colon"],
        }

    def test_column_order_is_kept(self, go_df):
        assert list(_implode(go_df, _make_args()).columns) == ["gene", "v", "tissue"]

    def test_rows_differing_elsewhere_are_not_joined(self):
        df = pd.DataFrame({"g": ["A", "A"], "v": ["x", "y"], "w": [1, 2]})
        assert _implode(df, _make_args())["v"].tolist() == ["x", "y"]

    def test_indexcols_groups_and_drops_other_columns(self):
        df = pd.DataFrame({"g": ["A", "A", "B"], "v": ["x", "y", "z"], "w": [1, 2, 3]})
        result = _implode(df, _make_args(indexcols=["g"]))
        assert list(result.columns) == ["g", "v"]
        assert result["v"].tolist() == ["x,y", "z"]

    def test_groups_keep_order_of_first_row(self):
        df = pd.DataFrame({"g": ["B", "A", "B"], "v": ["x", "y", "z"]})
        result = _implode(df, _make_args())
        assert result["g"].tolist() == ["B", "A"]
        assert result["v"].tolist() == ["x,z", "y"]

    def test_empty_key_forms_its_own_group(self):
        df = pd.DataFrame({"g": ["A", None, None], "v": ["x", "y", "z"]})
        result = _implode(df, _make_args())
        assert _values(result["g"]) == ["A", None]
        assert result["v"].tolist() == ["x", "y,z"]

    def test_only_joined_columns_gives_one_row(self):
        df = pd.DataFrame({"v": ["a", "b", "c"]})
        assert _implode(df, _make_args())["v"].tolist() == ["a,b,c"]

    def test_other_separator(self, go_df):
        result = _implode(go_df, _make_args(sep=" | "))
        assert result["v"].tolist() == ["GO:1 | GO:2", "GO:3"]

    def test_input_frame_is_not_modified(self, go_df):
        before = go_df.copy()
        _implode(go_df, _make_args())
        pd.testing.assert_frame_equal(go_df, before)


class TestImplodeValues:
    def test_empty_values_are_left_out(self):
        df = pd.DataFrame({"g": ["A", "A", "A"], "v": ["x", None, "y"]})
        assert _implode(df, _make_args())["v"].tolist() == ["x,y"]

    def test_group_without_values_gets_empty_cell(self):
        df = pd.DataFrame({"g": ["A", "B"], "v": ["x", None]})
        assert _values(_implode(df, _make_args())["v"]) == ["x", None]

    def test_unique_keeps_first_occurrence_order(self):
        df = pd.DataFrame({"g": ["A"] * 4, "v": ["y", "x", "y", "z"]})
        assert _implode(df, _make_args(unique=True))["v"].tolist() == ["y,x,z"]

    def test_sortvalues(self):
        df = pd.DataFrame({"g": ["A"] * 3, "v": ["y", "x", "z"]})
        assert _implode(df, _make_args(sortvalues=True))["v"].tolist() == ["x,y,z"]

    def test_sortvalues_sorts_numbers_numerically(self):
        df = pd.DataFrame({"g": ["A"] * 3, "v": [10, 9, 100]})
        assert _implode(df, _make_args(sortvalues=True))["v"].tolist() == ["9,10,100"]

    def test_whole_number_floats_lose_point_zero(self):
        # pandas reads an integer column with gaps as float64
        df = pd.DataFrame({"g": ["A", "A", "B"], "v": [2.0, 4.0, None]})
        assert _values(_implode(df, _make_args())["v"]) == ["2,4", None]

    def test_floats_keep_point_zero_if_any_value_is_fractional(self):
        df = pd.DataFrame({"g": ["A", "A"], "v": [1.5, 2.0]})
        assert _implode(df, _make_args())["v"].tolist() == ["1.5,2.0"]

    def test_nullable_integers(self):
        df = pd.DataFrame({"g": ["A", "A"], "v": pd.array([1, None], dtype="Int64")})
        assert _implode(df, _make_args())["v"].tolist() == ["1"]


class TestImplodeSeveralColumns:
    def test_columns_are_joined_in_the_same_order(self):
        df = pd.DataFrame({"id": [1, 1], "a": ["x", "y"], "b": [10, 20]})
        result = _implode(df, _make_args(cols=["a", "b"]))
        assert result["a"].tolist() == ["x,y"]
        assert result["b"].tolist() == ["10,20"]

    def test_unique_compares_whole_rows(self):
        df = pd.DataFrame({"id": [1, 1, 1], "a": ["x", "x", "y"], "b": [1, 2, 1]})
        result = _implode(df, _make_args(cols=["a", "b"], unique=True))
        assert result["a"].tolist() == ["x,x,y"]
        assert result["b"].tolist() == ["1,2,1"]

    def test_sortvalues_sorts_whole_rows(self):
        df = pd.DataFrame({"id": [1, 1, 1], "a": ["y", "x", "x"], "b": [1, 2, 1]})
        result = _implode(df, _make_args(cols=["a", "b"], sortvalues=True))
        assert result["a"].tolist() == ["x,x,y"]
        assert result["b"].tolist() == ["1,2,1"]


class TestImplodePoscol:
    def test_poscol_orders_values_and_is_dropped(self):
        df = pd.DataFrame({"id": [1, 1, 1], "v": ["y", "x", "z"], "n": [2, 1, 3]})
        result = _implode(df, _make_args(poscol="n"))
        assert list(result.columns) == ["id", "v"]
        assert result["v"].tolist() == ["x,y,z"]

    def test_poscol_is_not_a_grouping_column(self):
        df = pd.DataFrame({"id": [1, 1], "v": ["x", "y"], "n": [1, 2]})
        assert _implode(df, _make_args(poscol="n"))["v"].tolist() == ["x,y"]

    def test_round_trip_through_explode(self):
        df = pd.DataFrame(
            {"id": [1, 2], "a": ["x,y,z", "w"], "b": ["1,2,3", "4"], "c": ["p", "q"]}
        )
        exploded = _explode(
            df, make_args(cols=["a", "b"], sep=",", poscol="n", dropempty=False)
        )
        shuffled = exploded.sample(frac=1, random_state=3).reset_index(drop=True)
        result = _implode(shuffled, _make_args(cols=["a", "b"], poscol="n"))
        result = result.sort_values("id").reset_index(drop=True)
        assert result.to_dict("list") == {
            "id": [1, 2],
            "a": ["x,y,z", "w"],
            "b": ["1,2,3", "4"],
            "c": ["p", "q"],
        }


class TestImplodeErrors:
    def test_column_listed_twice(self, go_df):
        with pytest.raises(ValueError, match="more than once"):
            _implode(go_df, _make_args(cols=["v", "v"]))

    def test_indexcols_overlapping_cols(self, go_df):
        with pytest.raises(ValueError, match="overlaps"):
            _implode(go_df, _make_args(indexcols=["gene", "v"]))

    def test_poscol_in_cols(self, go_df):
        with pytest.raises(ValueError, match="also in -c"):
            _implode(go_df, _make_args(poscol="v"))


# ---------------------------------------------------------------------------
# ImplodeCommand.execute() — end to end
# ---------------------------------------------------------------------------


class TestImplodeExecute:
    def test_tsv_output(self, go_df):
        out = _run(go_df)
        assert out.to_dict("list") == {
            "gene": ["BRCA1", "TP53"],
            "v": ["GO:1,GO:2", "GO:3"],
            "tissue": ["breast", "colon"],
        }

    def test_missing_column(self, go_df):
        with pytest.raises(ValueError, match="-c/--cols"):
            _run(go_df, cols=["nope"])

    def test_missing_indexcol(self, go_df):
        with pytest.raises(ValueError, match="-i/--indexcols"):
            _run(go_df, indexcols=["nope"])

    def test_missing_poscol(self, go_df):
        with pytest.raises(ValueError, match="--poscol"):
            _run(go_df, poscol="nope")


class TestImplodeEdgeCases:
    def test_inf_is_written_as_inf(self):
        df = pd.DataFrame({"g": ["A", "A"], "v": [float("inf"), 2.0]})
        assert _implode(df, _make_args())["v"].tolist() == ["inf,2.0"]

    def test_column_named_gid_is_kept(self):
        df = pd.DataFrame(
            {"g": ["A", "A", "B"], "_gid": [9, 9, 8], "v": ["x", "y", "z"]}
        )
        result = _implode(df, _make_args())
        assert result["_gid"].tolist() == [9, 8]
        assert result["v"].tolist() == ["x,y", "z"]

    def test_indexcols_with_poscol(self):
        df = pd.DataFrame(
            {"g": ["A"] * 3, "v": ["y", "x", "z"], "n": [2, 1, 3], "w": [0, 1, 2]}
        )
        result = _implode(df, _make_args(indexcols=["g"], poscol="n"))
        assert list(result.columns) == ["g", "v"]
        assert result["v"].tolist() == ["x,y,z"]

    def test_column_with_only_empty_values(self):
        df = pd.DataFrame({"g": ["A", "B"], "v": [None, None]})
        assert _values(_implode(df, _make_args())["v"]) == [None, None]
