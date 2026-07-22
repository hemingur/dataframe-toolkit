"""Tests for dftk.commands.wstat_cmd._wstat."""

import pandas as pd
import pytest
import statsmodels.stats.weightstats as ssw

from dftk.commands.wstat_cmd import _wstat
from tests.conftest import make_args


def _args(**kwargs):
    defaults = dict(cols=["x"], weights=["w"], groups=[], confidencelevel=95.0)
    defaults.update(kwargs)
    return make_args(**defaults)


@pytest.fixture
def simple_df():
    return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "w": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def grouped_df():
    return pd.DataFrame(
        {
            "g": ["A", "A", "B", "B"],
            "x": [1.0, 3.0, 2.0, 4.0],
            "w": [1.0, 3.0, 1.0, 1.0],
        }
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_columns_basic(self, simple_df):
        result = _wstat(simple_df, _args())
        assert list(result.columns) == [
            "name",
            "weight",
            "totalweight",
            "wsum",
            "wmean",
            "wstd",
            "wmcilo",
            "wmcihi",
        ]

    def test_one_row_per_col(self, simple_df):
        result = _wstat(simple_df, _args())
        assert len(result) == 1

    def test_name_column(self, simple_df):
        result = _wstat(simple_df, _args())
        assert result["name"].iloc[0] == "x"

    def test_weight_column_name(self, simple_df):
        result = _wstat(simple_df, _args())
        assert result["weight"].iloc[0] == "w"

    def test_multi_col_rows(self, simple_df):
        df = simple_df.copy()
        df["y"] = df["x"] * 2
        result = _wstat(df, _args(cols=["x", "y"], weights=["w"]))
        assert len(result) == 2

    def test_grouped_columns_prepended(self, grouped_df):
        result = _wstat(grouped_df, _args(groups=["g"]))
        assert list(result.columns[:1]) == ["g"]
        assert "wmean" in result.columns

    def test_grouped_two_rows(self, grouped_df):
        result = _wstat(grouped_df, _args(groups=["g"]))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Correctness vs statsmodels
# ---------------------------------------------------------------------------


class TestCorrectness:
    def test_wmean_matches_statsmodels(self, simple_df):
        ws = ssw.DescrStatsW(simple_df["x"], simple_df["w"])
        result = _wstat(simple_df, _args())
        assert result["wmean"].iloc[0] == pytest.approx(ws.mean)

    def test_wstd_matches_statsmodels(self, simple_df):
        ws = ssw.DescrStatsW(simple_df["x"], simple_df["w"])
        result = _wstat(simple_df, _args())
        assert result["wstd"].iloc[0] == pytest.approx(ws.std)

    def test_totalweight(self, simple_df):
        result = _wstat(simple_df, _args())
        assert result["totalweight"].iloc[0] == pytest.approx(simple_df["w"].sum())

    def test_wsum(self, simple_df):
        expected = (simple_df["x"] * simple_df["w"]).sum()
        result = _wstat(simple_df, _args())
        assert result["wsum"].iloc[0] == pytest.approx(expected)

    def test_wmcilo_le_wmcihi(self, simple_df):
        result = _wstat(simple_df, _args())
        assert result["wmcilo"].iloc[0] <= result["wmcihi"].iloc[0]

    def test_uniform_weights_mean_equals_unweighted(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "w": [1.0, 1.0, 1.0, 1.0]})
        result = _wstat(df, _args())
        assert result["wmean"].iloc[0] == pytest.approx(df["x"].mean())

    def test_confidence_level_affects_ci(self, simple_df):
        r95 = _wstat(simple_df, _args(confidencelevel=95.0))
        r50 = _wstat(simple_df, _args(confidencelevel=50.0))
        width95 = r95["wmcihi"].iloc[0] - r95["wmcilo"].iloc[0]
        width50 = r50["wmcihi"].iloc[0] - r50["wmcilo"].iloc[0]
        assert width95 >= width50


# ---------------------------------------------------------------------------
# Weight broadcasting
# ---------------------------------------------------------------------------


class TestWeightBroadcast:
    def test_single_weight_broadcast(self):
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "w": [1.0, 2.0]})
        result = _wstat(df, _args(cols=["x", "y"], weights=["w"]))
        assert len(result) == 2
        assert list(result["weight"]) == ["w", "w"]

    def test_per_col_weights(self):
        df = pd.DataFrame(
            {"x": [1.0, 2.0], "y": [3.0, 4.0], "w1": [1.0, 2.0], "w2": [2.0, 1.0]}
        )
        result = _wstat(df, _args(cols=["x", "y"], weights=["w1", "w2"]))
        assert list(result["weight"]) == ["w1", "w2"]

    def test_mismatched_weights_raises(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0], "w1": [1.0], "w2": [1.0]})
        with pytest.raises(ValueError, match="must match"):
            _wstat(df, _args(cols=["x", "y"], weights=["w1", "w2", "w1"]))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouping:
    def test_group_values(self, grouped_df):
        result = _wstat(grouped_df, _args(groups=["g"]))
        assert set(result["g"]) == {"A", "B"}

    def test_group_a_wmean(self, grouped_df):
        # group A: x=[1,3], w=[1,3] → wmean = (1*1 + 3*3)/(1+3) = 10/4 = 2.5
        result = _wstat(grouped_df, _args(groups=["g"]))
        row_a = result[result["g"] == "A"]
        assert row_a["wmean"].iloc[0] == pytest.approx(2.5)

    def test_multi_group_col(self):
        df = pd.DataFrame(
            {
                "g1": ["X", "X", "Y", "Y"],
                "g2": ["P", "Q", "P", "Q"],
                "x": [1.0, 2.0, 3.0, 4.0],
                "w": [1.0, 1.0, 1.0, 1.0],
            }
        )
        result = _wstat(df, _args(groups=["g1", "g2"]))
        assert list(result.columns[:2]) == ["g1", "g2"]
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row(self):
        df = pd.DataFrame({"x": [5.0], "w": [2.0]})
        result = _wstat(df, _args())
        assert result["wmean"].iloc[0] == pytest.approx(5.0)

    def test_nan_in_data_excluded(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0], "w": [1.0, 1.0, 1.0]})
        result = _wstat(df, _args())
        assert result["totalweight"].iloc[0] == pytest.approx(2.0)
        assert result["wmean"].iloc[0] == pytest.approx(2.0)

    def test_nan_in_weight_excluded(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "w": [1.0, float("nan"), 1.0]})
        result = _wstat(df, _args())
        assert result["totalweight"].iloc[0] == pytest.approx(2.0)
