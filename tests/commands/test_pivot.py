"""
Tests for stattools.commands.pivot_cmd._do_pivot and _prepare_funcs.
"""

import numpy as np
import pandas as pd
import pytest

from stattools.commands.pivot_cmd import _do_pivot, _flatten_columns, _prepare_funcs
from tests.conftest import make_args

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def long_df():
    """Simple long-format data: 2 regions × 2 products, one sales value each."""
    return pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "product": ["A", "B", "A", "B"],
            "sales": [10.0, 20.0, 30.0, 40.0],
        }
    )


@pytest.fixture
def two_value_df():
    """Two value columns for testing multi-value pivots."""
    return pd.DataFrame(
        {
            "region": ["N", "N", "S", "S"],
            "product": ["A", "B", "A", "B"],
            "sales": [10.0, 20.0, 30.0, 40.0],
            "cost": [1.0, 2.0, 3.0, 4.0],
        }
    )


@pytest.fixture
def multi_row_df():
    """Multiple rows per (region, product) cell for testing aggregation."""
    return pd.DataFrame(
        {
            "region": ["N", "N", "N", "N", "S", "S", "S", "S"],
            "product": ["A", "A", "B", "B", "A", "A", "B", "B"],
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )


def _pivot_args(**overrides):
    defaults = dict(
        values=["sales"],
        index=["region"],
        groupcols=["product"],
        aggfunc=None,
        fillzero=False,
        confidencelevel=95.0,
        confidencemethod="linear",
    )
    defaults.update(overrides)
    return make_args(**defaults)


# ---------------------------------------------------------------------------
# _flatten_columns
# ---------------------------------------------------------------------------


class TestFlattenColumns:
    def test_plain_strings_unchanged(self):
        assert _flatten_columns(["a", "b"]) == ["a", "b"]

    def test_tuples_joined(self):
        assert _flatten_columns([("mean", "x"), ("std", "x")]) == ["mean_x", "std_x"]

    def test_empty_parts_skipped(self):
        # pandas sometimes produces ('', 'A') for single-value cases
        assert _flatten_columns([("", "A"), ("", "B")]) == ["A", "B"]

    def test_mixed(self):
        assert _flatten_columns(["region", ("sales", "A")]) == ["region", "sales_A"]


# ---------------------------------------------------------------------------
# _prepare_funcs
# ---------------------------------------------------------------------------


class TestPrepareFuncs:
    def test_none_returns_none(self):
        assert _prepare_funcs(None, 95.0, "linear") is None

    def test_bare_names_returns_list(self):
        result = _prepare_funcs(["mean", "std"], 95.0, "linear")
        assert result == ["mean", "std"]

    def test_col_func_syntax_returns_dict(self):
        result = _prepare_funcs(["sales:mean", "cost:sum"], 95.0, "linear")
        assert isinstance(result, dict)
        assert result["sales"] == ["mean"]
        assert result["cost"] == ["sum"]

    def test_cilo_cihi_named_correctly(self):
        result = _prepare_funcs(["cilo", "cihi"], 95.0, "linear")
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].__name__ == "cilo"
        assert result[1].__name__ == "cihi"

    def test_cilo_level_95(self):
        funcs = _prepare_funcs(["cilo"], 95.0, "linear")
        # cilo at 95% CI = 2.5th percentile of [1,2,3,4,5]
        val = funcs[0](np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        expected = np.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 2.5, method="linear")
        assert val == pytest.approx(expected)

    def test_mixed_syntax_raises(self):
        # COL:FUNC must come first; a bare item after that triggers the Mixed check
        with pytest.raises(ValueError, match="Mixed"):
            _prepare_funcs(["sales:mean", "std"], 95.0, "linear")

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            _prepare_funcs(["not_a_function"], 95.0, "linear")

    def test_bitwise_and(self):
        result = _prepare_funcs(["bitwise_and"], 95.0, "linear")
        assert isinstance(result, list)
        assert result[0](np.array([0b1100, 0b1010])) == 0b1000

    def test_numpy_function(self):
        # numpy functions resolved via getattr(np, name)
        result = _prepare_funcs(["nanmean"], 95.0, "linear")
        assert isinstance(result, list)
        assert result[0]([1.0, np.nan, 3.0]) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# _do_pivot — basic behaviour
# ---------------------------------------------------------------------------


class TestDoPivotBasic:
    def test_schema_has_index_and_group_cols(self, long_df):
        result = _do_pivot(long_df, _pivot_args())
        assert "region" in result.columns
        assert len(result) == 2  # N and S

    def test_group_column_values_become_headers(self, long_df):
        result = _do_pivot(long_df, _pivot_args())
        # product A and B should become column names (possibly prefixed)
        col_str = " ".join(result.columns)
        assert "A" in col_str
        assert "B" in col_str

    def test_north_values(self, long_df):
        # values as a list → MultiIndex cols → flatten gives e.g. "sales_A"
        result = _do_pivot(long_df, _pivot_args())
        n_row = result[result["region"] == "N"].iloc[0]
        col_a = [c for c in result.columns if c.endswith("A")][0]
        col_b = [c for c in result.columns if c.endswith("B")][0]
        assert n_row[col_a] == pytest.approx(10.0)
        assert n_row[col_b] == pytest.approx(20.0)

    def test_south_values(self, long_df):
        result = _do_pivot(long_df, _pivot_args())
        s_row = result[result["region"] == "S"].iloc[0]
        col_a = [c for c in result.columns if c.endswith("A")][0]
        col_b = [c for c in result.columns if c.endswith("B")][0]
        assert s_row[col_a] == pytest.approx(30.0)
        assert s_row[col_b] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# _do_pivot — aggregation
# ---------------------------------------------------------------------------


class TestDoPivotAggregation:
    def test_mean_aggregation(self, multi_row_df):
        result = _do_pivot(multi_row_df, _pivot_args(aggfunc=["mean"]))
        n_row = result[result["region"] == "N"].iloc[0]
        # N/A: mean([10, 20]) = 15; N/B: mean([30, 40]) = 35
        col_a = [c for c in result.columns if c.endswith("A")][0]
        col_b = [c for c in result.columns if c.endswith("B")][0]
        assert n_row[col_a] == pytest.approx(15.0)
        assert n_row[col_b] == pytest.approx(35.0)

    def test_multiple_aggfuncs(self, multi_row_df):
        result = _do_pivot(multi_row_df, _pivot_args(aggfunc=["mean", "std"]))
        mean_cols = [c for c in result.columns if "mean" in c]
        std_cols = [c for c in result.columns if "std" in c]
        assert len(mean_cols) > 0
        assert len(std_cols) > 0

    def test_two_value_columns(self, two_value_df):
        result = _do_pivot(two_value_df, _pivot_args(values=["sales", "cost"]))
        sales_cols = [c for c in result.columns if "sales" in c]
        cost_cols = [c for c in result.columns if "cost" in c]
        assert len(sales_cols) > 0
        assert len(cost_cols) > 0


# ---------------------------------------------------------------------------
# _do_pivot — fillzero
# ---------------------------------------------------------------------------


class TestDoPivotFillZero:
    def test_fillzero_completes_sparse_table(self):
        """With a sparse table and --fillzero, missing cells become 0."""
        df = pd.DataFrame(
            {
                "region": ["N", "S"],
                "product": ["A", "B"],  # only diagonal is populated
                "sales": [10.0, 40.0],
            }
        )
        result = _do_pivot(df, _pivot_args(fillzero=True))
        # After fillzero both N and S rows should exist
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Bootstrap integration
# ---------------------------------------------------------------------------


class TestBootstrap:
    def test_bootstrap_produces_single_df(self, long_df):
        """Bootstrap should emit one concatenated DataFrame, not multiple blocks."""
        import argparse
        import sys

        from stattools.commands.pivot_cmd import PivotCommand

        cmd = PivotCommand()
        parser = argparse.ArgumentParser()
        cmd.add_arguments(parser)
        args = parser.parse_args(
            [
                "-v",
                "sales",
                "-i",
                "region",
                "-g",
                "product",
                "-f",
                "mean",
                "--bootstrap",
                "5",
                "--randomseed",
                "42",
                "--fullsampling",
            ]
        )
        args.DATAFILE = None

        # Redirect stdout to capture TSV output
        import io as _io

        buf = _io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        # Monkey-patch io.read to return long_df directly
        from stattools.common import io as common_io

        original_read = common_io.io.read
        common_io.io.read = lambda a: long_df

        try:
            cmd.execute(args)
        finally:
            sys.stdout = old_stdout
            common_io.io.read = original_read

        buf.seek(0)
        result = pd.read_csv(buf, sep="\t")
        assert "samplenum" in result.columns
        assert set(result["samplenum"]) == {1, 2, 3, 4, 5}
