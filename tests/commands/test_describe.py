"""Tests for stattools.commands.describe_cmd."""

import io as _io
import math
import sys

import numpy as np
import pandas as pd
import pytest

from stattools.commands.describe_cmd import (
    DescribeCommand,
    _col_type,
    _correlations,
    _generate_summary,
    _profile_col,
)
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(summary=False, correlations=False, corr_threshold=0.7)
    defaults.update(kwargs)
    return make_args(**defaults)


def _run(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run DescribeCommand on *df* and return the profile DataFrame."""
    import stattools.commands.describe_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    buf = _io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        DescribeCommand().execute(_make_args(**kwargs))
    finally:
        sys.stdout = old_stdout
        mod.io.read = original_read
    return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")


# ---------------------------------------------------------------------------
# _col_type
# ---------------------------------------------------------------------------


class TestColType:
    def test_numeric_float(self):
        assert _col_type(pd.Series([1.0, 2.0, 3.0])) == "numeric"

    def test_numeric_int(self):
        assert _col_type(pd.Series([1, 2, 3])) == "numeric"

    def test_boolean(self):
        assert _col_type(pd.Series([True, False, True])) == "boolean"

    def test_categorical(self):
        assert _col_type(pd.Series(["a", "b", "c"])) == "categorical"

    def test_datetime(self):
        s = pd.Series(pd.to_datetime(["2020-01-01", "2020-06-01"]))
        assert _col_type(s) == "datetime"


# ---------------------------------------------------------------------------
# _profile_col — basic fields
# ---------------------------------------------------------------------------


class TestProfileColBasic:
    def test_numeric_fields_present(self):
        col = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
        row = _profile_col(col, 5)
        assert row["name"] == "x"
        assert row["type"] == "numeric"
        assert row["count"] == 5
        assert row["missing_pct"] == 0.0
        assert row["n_unique"] == 5
        assert row["mean"] == pytest.approx(3.0)
        assert row["std"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
        assert row["min"] == 1.0
        assert row["max"] == 5.0
        assert row["median"] == 3.0

    def test_categorical_fields(self):
        col = pd.Series(["a", "b", "a", "a"], name="cat")
        row = _profile_col(col, 4)
        assert row["type"] == "categorical"
        assert row["top"] == "a"
        assert row["top_freq_pct"] == pytest.approx(75.0)
        assert math.isnan(row["mean"])

    def test_missing_count(self):
        col = pd.Series([1.0, np.nan, 3.0, np.nan], name="x")
        row = _profile_col(col, 4)
        assert row["count"] == 2
        assert row["missing_pct"] == 50.0

    def test_all_missing(self):
        col = pd.Series([np.nan, np.nan], name="x")
        row = _profile_col(col, 2)
        assert "all_missing" in row["notes"]
        assert "high_missing" not in row["notes"]  # all_missing takes precedence

    def test_high_missing_flag(self):
        col = pd.Series([1.0] + [np.nan] * 4, name="x")
        row = _profile_col(col, 5)
        assert "high_missing" in row["notes"]

    def test_no_high_missing_below_threshold(self):
        col = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan], name="x")
        row = _profile_col(col, 5)
        assert "high_missing" not in row["notes"]


# ---------------------------------------------------------------------------
# _profile_col — notes / flags
# ---------------------------------------------------------------------------


class TestProfileColNotes:
    def test_constant_flag(self):
        col = pd.Series([7.0, 7.0, 7.0], name="x")
        row = _profile_col(col, 3)
        assert "constant" in row["notes"]

    def test_possible_id_flag(self):
        col = pd.Series([1, 2, 3, 4, 5], name="id")
        row = _profile_col(col, 5)
        assert "possible_id" in row["notes"]

    def test_no_possible_id_with_duplicates(self):
        col = pd.Series([1, 2, 2, 3], name="x")
        row = _profile_col(col, 4)
        assert "possible_id" not in row["notes"]

    def test_approx_normal_flag(self):
        rng = np.random.default_rng(42)
        col = pd.Series(rng.normal(0, 1, 500), name="x")
        row = _profile_col(col, 500)
        assert "approx_normal" in row["notes"]

    def test_right_skewed_flag(self):
        # Exponential distribution is strongly right-skewed
        rng = np.random.default_rng(42)
        col = pd.Series(rng.exponential(1.0, 500), name="x")
        row = _profile_col(col, 500)
        assert "right_skewed" in row["notes"]

    def test_left_skewed_flag(self):
        # Negate an exponential to get left skew
        rng = np.random.default_rng(42)
        col = pd.Series(-rng.exponential(1.0, 500), name="x")
        row = _profile_col(col, 500)
        assert "left_skewed" in row["notes"]

    def test_outliers_iqr_flag(self):
        # Normal data plus one extreme outlier
        vals = [1.0, 2.0, 3.0, 2.5, 2.0, 1.5, 2.2, 1000.0]
        col = pd.Series(vals, name="x")
        row = _profile_col(col, len(vals))
        assert "outliers_iqr" in row["notes"]

    def test_no_outliers_flag_clean_data(self):
        # Uniform integers 0-9 repeated — IQR fences are wide, no outliers
        col = pd.Series([float(x) for x in list(range(10)) * 3], name="x")
        row = _profile_col(col, 30)
        assert "outliers_iqr" not in row["notes"]

    def test_near_constant_categorical(self):
        # One value dominates > 95 %
        col = pd.Series(["a"] * 99 + ["b"], name="cat")
        row = _profile_col(col, 100)
        assert "near_constant" in row["notes"]

    def test_high_cardinality_categorical(self):
        col = pd.Series([str(i) for i in range(100)], name="cat")
        row = _profile_col(col, 100)
        assert "high_cardinality" in row["notes"]

    def test_no_high_cardinality_small(self):
        col = pd.Series(["a", "b", "c"] * 10, name="cat")
        row = _profile_col(col, 30)
        assert "high_cardinality" not in row["notes"]


# ---------------------------------------------------------------------------
# _correlations
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        pairs = _correlations(df, threshold=0.7)
        assert len(pairs) == 1
        a, b, r = pairs[0]
        assert {a, b} == {"x", "y"}
        assert r == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [-1.0, -2.0, -3.0]})
        pairs = _correlations(df, threshold=0.7)
        assert len(pairs) == 1
        assert abs(pairs[0][2]) == pytest.approx(1.0)

    def test_below_threshold_excluded(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 50), "y": rng.normal(0, 1, 50)})
        pairs = _correlations(df, threshold=0.7)
        assert all(abs(r) >= 0.7 for _, _, r in pairs)

    def test_sorted_by_magnitude(self):
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [1.0, 2.0, 3.0, 4.0],  # r=1.0 with a
                "c": [1.0, 2.1, 2.9, 4.2],  # r≈0.999 with a
            }
        )
        pairs = _correlations(df, threshold=0.7)
        rs = [abs(r) for _, _, r in pairs]
        assert rs == sorted(rs, reverse=True)

    def test_no_numeric_columns(self):
        df = pd.DataFrame({"x": ["a", "b"], "y": ["c", "d"]})
        assert _correlations(df) == []

    def test_single_numeric_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "cat": ["a", "b"]})
        assert _correlations(df) == []

    def test_threshold_respected(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        assert len(_correlations(df, threshold=0.99)) == 1
        assert len(_correlations(df, threshold=1.01)) == 0


# ---------------------------------------------------------------------------
# _generate_summary
# ---------------------------------------------------------------------------


class TestGenerateSummary:
    def _profile(self, notes_map: dict) -> pd.DataFrame:
        rows = []
        for name, (col_type, notes) in notes_map.items():
            rows.append(
                {"name": name, "type": col_type, "missing_pct": 0.0, "notes": notes}
            )
        return pd.DataFrame(rows)

    def test_dataset_dimensions(self):
        profile = self._profile({"x": ("numeric", "")})
        text = _generate_summary(profile, [], 100, 3)
        assert "100 rows" in text
        assert "3 columns" in text

    def test_missing_data_mentioned(self):
        rows = [
            {
                "name": "x",
                "type": "numeric",
                "missing_pct": 15.0,
                "notes": "high_missing",
            }
        ]
        profile = pd.DataFrame(rows)
        text = _generate_summary(profile, [], 100, 1)
        assert "x" in text
        assert "15.0%" in text

    def test_possible_id_mentioned(self):
        profile = self._profile({"user_id": ("categorical", "possible_id")})
        text = _generate_summary(profile, [], 10, 1)
        assert "user_id" in text
        assert "identifier" in text.lower()

    def test_distribution_note_right_skewed(self):
        profile = self._profile({"income": ("numeric", "right_skewed,outliers_iqr")})
        text = _generate_summary(profile, [], 100, 1)
        assert "income" in text
        assert "right-skewed" in text

    def test_correlations_mentioned(self):
        profile = self._profile({"x": ("numeric", ""), "y": ("numeric", "")})
        text = _generate_summary(profile, [("x", "y", 0.95)], 10, 2)
        assert "x" in text and "y" in text
        assert "0.95" in text

    def test_no_correlations_section_when_empty(self):
        profile = self._profile({"x": ("numeric", "")})
        text = _generate_summary(profile, [], 10, 1)
        assert "correlation" not in text.lower()


# ---------------------------------------------------------------------------
# DescribeCommand.execute — integration
# ---------------------------------------------------------------------------


class TestDescribeExecute:
    def test_one_row_per_column(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": ["a", "b"], "z": [True, False]})
        result = _run(df)
        assert len(result) == 3
        assert list(result["name"]) == ["x", "y", "z"]

    def test_output_columns_present(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = _run(df)
        for col in [
            "name",
            "type",
            "count",
            "missing_pct",
            "n_unique",
            "mean",
            "std",
            "min",
            "p25",
            "median",
            "p75",
            "max",
            "skew",
            "kurtosis",
            "top",
            "top_freq_pct",
            "notes",
        ]:
            assert col in result.columns, f"Missing column: {col}"

    def test_numeric_stats_correct(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = _run(df)
        row = result[result["name"] == "x"].iloc[0]
        assert float(row["mean"]) == pytest.approx(3.0)
        assert float(row["min"]) == pytest.approx(1.0)
        assert float(row["max"]) == pytest.approx(5.0)

    def test_categorical_type(self):
        df = pd.DataFrame({"cat": ["a", "b", "a"]})
        result = _run(df)
        assert result.iloc[0]["type"] == "categorical"
        assert result.iloc[0]["top"] == "a"

    def test_possible_id_detected(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        result = _run(df)
        id_row = result[result["name"] == "id"].iloc[0]
        assert "possible_id" in str(id_row["notes"])

    def test_summary_goes_to_stderr(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _run(df, summary=True)
        captured = capsys.readouterr()
        assert "rows" in captured.err
        assert "columns" in captured.err

    def test_correlations_in_stderr(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        _run(df, summary=True, correlations=True)
        captured = capsys.readouterr()
        assert "correlation" in captured.err.lower()
        assert "x" in captured.err and "y" in captured.err

    def test_no_stderr_without_flags(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        _run(df)
        assert capsys.readouterr().err == ""

    def test_empty_dataframe(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = _run(df)
        assert len(result) == 1
        assert result.iloc[0]["count"] == 0

    def test_all_missing_column(self):
        df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})
        result = _run(df)
        assert "all_missing" in str(result.iloc[0]["notes"])
