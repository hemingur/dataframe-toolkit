"""
Tests for stattools.commands.stat._compute_stats.

Strategy: synthetic datasets with exactly known properties.  Expected values
are computed from numpy/pandas directly so the tests verify the function's
*orchestration* (correct aggregation, schema, reshape), not reimplementations
of numpy math.
"""

import numpy as np
import pandas as pd
import pytest

from stattools.commands.stat_cmd import _STAT_NAMES, _compute_stats
from tests.conftest import make_args

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_OUTPUT_COLS = ["name"] + _STAT_NAMES


def _stat_args(**overrides):
    defaults = dict(
        cols=["x"],
        groupcol=None,
        confidencelevel=95.0,
        confidencemethod="linear",
    )
    defaults.update(overrides)
    return make_args(**defaults)


def _row(result: pd.DataFrame, name: str) -> pd.Series:
    """Return the single row where result["name"] == name."""
    rows = result[result["name"] == name]
    assert len(rows) == 1, (
        f"Expected exactly one row with name={name!r}, got {len(rows)}"
    )
    return rows.iloc[0]


def _group_row(result: pd.DataFrame, group_col: str, group_val, name: str) -> pd.Series:
    """Return the single row for a given group value and stat name."""
    mask = (result[group_col] == group_val) & (result["name"] == name)
    rows = result[mask]
    assert len(rows) == 1
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# Ungrouped — single column
# ---------------------------------------------------------------------------


class TestUngroupedSingleCol:
    """_compute_stats with groupcol=None and a single column."""

    def test_schema(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert list(result.columns) == _ALL_OUTPUT_COLS

    def test_one_row(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert len(result) == 1

    def test_name_column(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["name"].iloc[0] == "x"

    def test_count(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["count"].iloc[0] == 5

    def test_sum(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["sum"].iloc[0] == pytest.approx(15.0)

    def test_mean(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["mean"].iloc[0] == pytest.approx(3.0)

    def test_std(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        expected = simple_df["x"].std()  # pandas default ddof=1
        assert result["std"].iloc[0] == pytest.approx(expected)

    def test_min_max(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["min"].iloc[0] == pytest.approx(1.0)
        assert result["max"].iloc[0] == pytest.approx(5.0)

    def test_median(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["median"].iloc[0] == pytest.approx(3.0)

    def test_sem(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        expected = simple_df["x"].sem()
        assert result["sem"].iloc[0] == pytest.approx(expected)

    def test_skew_symmetric(self, simple_df):
        # [1,2,3,4,5] is perfectly symmetric → skew == 0
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert result["skew"].iloc[0] == pytest.approx(0.0, abs=1e-10)

    def test_kurt(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        expected = simple_df["x"].kurt()
        assert result["kurt"].iloc[0] == pytest.approx(expected)

    def test_ci_95_default(self, simple_df):
        """cilo/cihi match np.percentile at 2.5 and 97.5."""
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        x = simple_df["x"].to_numpy()
        expected_lo = np.percentile(x, 2.5, method="linear")
        expected_hi = np.percentile(x, 97.5, method="linear")
        assert result["cilo"].iloc[0] == pytest.approx(expected_lo)
        assert result["cihi"].iloc[0] == pytest.approx(expected_hi)

    def test_ci_99(self, simple_df):
        """Custom confidence level is forwarded correctly."""
        result = _compute_stats(simple_df, _stat_args(cols=["x"], confidencelevel=99.0))
        x = simple_df["x"].to_numpy()
        expected_lo = np.percentile(x, 0.5, method="linear")
        expected_hi = np.percentile(x, 99.5, method="linear")
        assert result["cilo"].iloc[0] == pytest.approx(expected_lo)
        assert result["cihi"].iloc[0] == pytest.approx(expected_hi)


# ---------------------------------------------------------------------------
# Ungrouped — multiple columns
# ---------------------------------------------------------------------------


class TestUngroupedMultiCol:
    """_compute_stats always emits a 'name' column when multiple cols requested."""

    def test_schema(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x", "y"]))
        assert list(result.columns) == _ALL_OUTPUT_COLS

    def test_two_rows(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x", "y"]))
        assert len(result) == 2

    def test_name_values(self, simple_df):
        result = _compute_stats(simple_df, _stat_args(cols=["x", "y"]))
        assert set(result["name"]) == {"x", "y"}

    def test_y_mean_is_10x_x(self, simple_df):
        """y == 10*x throughout, so stats should scale accordingly."""
        result = _compute_stats(simple_df, _stat_args(cols=["x", "y"]))
        x_row = _row(result, "x")
        y_row = _row(result, "y")
        assert y_row["mean"] == pytest.approx(10 * x_row["mean"])
        assert y_row["sum"] == pytest.approx(10 * x_row["sum"])
        assert y_row["std"] == pytest.approx(10 * x_row["std"])


# ---------------------------------------------------------------------------
# Grouped — single column
# ---------------------------------------------------------------------------


class TestGroupedSingleCol:
    def test_schema(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        assert list(result.columns) == ["group", "name"] + _STAT_NAMES

    def test_two_rows(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        assert len(result) == 2  # one row per (group, col) combination

    def test_group_names(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        assert set(result["group"]) == {"A", "B"}

    def test_group_a_mean(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        row = _group_row(result, "group", "A", "value")
        assert row["mean"] == pytest.approx(2.0)

    def test_group_b_mean(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        row = _group_row(result, "group", "B", "value")
        assert row["mean"] == pytest.approx(20.0)

    def test_group_a_count(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        row = _group_row(result, "group", "A", "value")
        assert row["count"] == 3

    def test_group_b_sum(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        row = _group_row(result, "group", "B", "value")
        assert row["sum"] == pytest.approx(60.0)

    def test_group_a_ci(self, grouped_df):
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        row = _group_row(result, "group", "A", "value")
        a_vals = np.array([1.0, 2.0, 3.0])
        expected_lo = np.percentile(a_vals, 2.5, method="linear")
        expected_hi = np.percentile(a_vals, 97.5, method="linear")
        assert row["cilo"] == pytest.approx(expected_lo)
        assert row["cihi"] == pytest.approx(expected_hi)

    def test_kurt_present(self, grouped_df):
        """Grouped path includes kurt (was missing in the legacy script)."""
        result = _compute_stats(
            grouped_df, _stat_args(cols=["value"], groupcol=["group"])
        )
        assert "kurt" in result.columns


# ---------------------------------------------------------------------------
# Grouped — multiple columns
# ---------------------------------------------------------------------------


class TestGroupedMultiCol:
    def test_schema(self, two_group_two_col_df):
        result = _compute_stats(
            two_group_two_col_df,
            _stat_args(cols=["x", "y"], groupcol=["group"]),
        )
        assert list(result.columns) == ["group", "name"] + _STAT_NAMES

    def test_four_rows(self, two_group_two_col_df):
        """2 groups × 2 columns = 4 rows."""
        result = _compute_stats(
            two_group_two_col_df,
            _stat_args(cols=["x", "y"], groupcol=["group"]),
        )
        assert len(result) == 4

    def test_name_values(self, two_group_two_col_df):
        result = _compute_stats(
            two_group_two_col_df,
            _stat_args(cols=["x", "y"], groupcol=["group"]),
        )
        assert set(result["name"]) == {"x", "y"}

    def test_group_a_x_mean(self, two_group_two_col_df):
        result = _compute_stats(
            two_group_two_col_df,
            _stat_args(cols=["x", "y"], groupcol=["group"]),
        )
        row = _group_row(result, "group", "A", "x")
        assert row["mean"] == pytest.approx(2.0)  # mean([1, 3])

    def test_group_b_y_mean(self, two_group_two_col_df):
        result = _compute_stats(
            two_group_two_col_df,
            _stat_args(cols=["x", "y"], groupcol=["group"]),
        )
        row = _group_row(result, "group", "B", "y")
        assert row["mean"] == pytest.approx(30.0)  # mean([20, 40])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row(self):
        """Stats for a single-row DataFrame should not crash."""
        df = pd.DataFrame({"x": [42.0]})
        result = _compute_stats(df, _stat_args(cols=["x"]))
        assert result["count"].iloc[0] == 1
        assert result["mean"].iloc[0] == pytest.approx(42.0)

    def test_name_col_clash_warning(self, caplog):
        """If a group column is named 'name', stat renames to 'name_mangled'."""
        df = pd.DataFrame({"name": ["A", "A", "B"], "value": [1.0, 2.0, 3.0]})
        import logging

        with caplog.at_level(logging.WARNING, logger="stattools"):
            result = _compute_stats(df, _stat_args(cols=["value"], groupcol=["name"]))
        assert "name_mangled" in result.columns
        assert "name" in caplog.text

    def test_ci_method_nearest(self, simple_df):
        """confidencemethod is forwarded to np.percentile."""
        result = _compute_stats(
            simple_df,
            _stat_args(cols=["x"], confidencemethod="nearest"),
        )
        x = simple_df["x"].to_numpy()
        expected_lo = np.percentile(x, 2.5, method="nearest")
        assert result["cilo"].iloc[0] == pytest.approx(expected_lo)

    def test_stat_names_order(self, simple_df):
        """Output columns follow the canonical _STAT_NAMES order."""
        result = _compute_stats(simple_df, _stat_args(cols=["x"]))
        assert list(result.columns[1:]) == _STAT_NAMES
