"""
Tests for stattools.commands.scale_cmd.
"""

import argparse

import numpy as np
import pandas as pd
import pytest

from stattools.commands.scale_cmd import ScaleCommand, rankcols, residcols, scalecols

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    return pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 20.0, 30.0, 40.0, 50.0]}
    )


@pytest.fixture
def grouped_df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "x": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def regression_df():
    rng = np.random.default_rng(0)
    x = np.arange(20, dtype=float)
    y = 3.0 * x + 5.0 + rng.normal(0, 0.5, 20)
    return pd.DataFrame({"x": x, "y": y})


def _args(**kwargs) -> argparse.Namespace:
    defaults = dict(
        cols=["x"],
        groupcol=None,
        rank=False,
        resid=False,
        shift="mean",
        scale_by="std",
        rankdist="normal",
        formula=None,
        nointercept=False,
        verbose=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# scalecols — shift options
# ---------------------------------------------------------------------------


class TestScaleColsShift:
    def test_shift_mean_gives_zero_mean(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(scale_by="none"))
        assert result["x_scaled"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_shift_min_gives_zero_minimum(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="min", scale_by="none"))
        assert result["x_scaled"].min() == pytest.approx(0.0)

    def test_shift_first_gives_zero_first(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="first", scale_by="none"))
        assert result["x_scaled"].iloc[0] == pytest.approx(0.0)

    def test_shift_none_preserves_values(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="none", scale_by="none"))
        np.testing.assert_array_almost_equal(
            result["x_scaled"].values, simple_df["x"].values
        )


# ---------------------------------------------------------------------------
# scalecols — scale options
# ---------------------------------------------------------------------------


class TestScaleColsScale:
    def test_scale_std_gives_unit_std(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args())
        assert result["x_scaled"].std() == pytest.approx(1.0, abs=0.01)

    def test_scale_sum_sums_to_one(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="none", scale_by="sum"))
        assert result["x_scaled"].sum() == pytest.approx(1.0)

    def test_scale_max_max_is_one(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="none", scale_by="max"))
        assert result["x_scaled"].abs().max() == pytest.approx(1.0)

    def test_scale_mean_mean_is_one(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(shift="none", scale_by="mean"))
        assert result["x_scaled"].mean() == pytest.approx(1.0)

    def test_scale_none_no_division(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(scale_by="none"))
        # after mean shift, values should be [-2, -1, 0, 1, 2]
        np.testing.assert_array_almost_equal(
            result["x_scaled"].values, [-2.0, -1.0, 0.0, 1.0, 2.0]
        )

    def test_zscore_default(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args())
        assert result["x_scaled"].mean() == pytest.approx(0.0, abs=1e-10)
        assert result["x_scaled"].std() == pytest.approx(1.0, abs=0.01)

    def test_multiple_cols(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args(cols=["x", "y"]))
        assert "x_scaled" in result.columns
        assert "y_scaled" in result.columns

    def test_original_cols_preserved(self, simple_df):
        df = simple_df.copy()
        result = scalecols(df, _args())
        assert "x" in result.columns

    def test_constant_column_no_crash(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
        result = scalecols(df, _args())
        # std=0, scale_val=0 → shifted / 0 not raised, returns shifted
        assert "x_scaled" in result.columns


# ---------------------------------------------------------------------------
# rankcols
# ---------------------------------------------------------------------------


class TestRankCols:
    def test_uniform_range(self, simple_df):
        df = simple_df.copy()
        result = rankcols(df, _args(rank=True, rankdist="uniform"))
        assert result["x_scaled"].min() >= 0.0
        assert result["x_scaled"].max() <= 1.0

    def test_uniform_monotone(self, simple_df):
        df = simple_df.copy()
        result = rankcols(df, _args(rank=True, rankdist="uniform"))
        vals = result["x_scaled"].values
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_normal_zero_mean(self, simple_df):
        df = simple_df.copy()
        result = rankcols(df, _args(rank=True, rankdist="normal"))
        assert result["x_scaled"].mean() == pytest.approx(0.0, abs=1e-10)

    def test_normal_unit_std(self, simple_df):
        df = simple_df.copy()
        result = rankcols(df, _args(rank=True, rankdist="normal"))
        assert result["x_scaled"].std(ddof=0) == pytest.approx(1.0, abs=1e-10)

    def test_normal_monotone(self, simple_df):
        df = simple_df.copy()
        result = rankcols(df, _args(rank=True, rankdist="normal"))
        vals = result["x_scaled"].values
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))

    def test_ties_handled(self):
        df = pd.DataFrame({"x": [1.0, 1.0, 2.0, 3.0]})
        result = rankcols(df, _args(rank=True, rankdist="uniform"))
        # tied values should get the same rank score
        assert result["x_scaled"].iloc[0] == pytest.approx(result["x_scaled"].iloc[1])


# ---------------------------------------------------------------------------
# residcols
# ---------------------------------------------------------------------------


class TestResidCols:
    def test_resid_col_created(self, regression_df):
        args = _args(resid=True, formula="y ~ x", cols=None)
        result = residcols(regression_df.copy(), args)
        assert "y_scaled" in result.columns

    def test_resid_mean_near_intercept(self, regression_df):
        """With nointercept=False, residuals should be re-centred at the intercept."""
        args = _args(resid=True, formula="y ~ x", cols=None)
        result = residcols(regression_df.copy(), args)
        # intercept ≈ 5, residuals ≈ 0, so y_scaled should be centered around 5
        assert result["y_scaled"].mean() == pytest.approx(5.0, abs=1.0)

    def test_resid_nointercept_zero_mean(self, regression_df):
        """Raw residuals should be centered around 0."""
        args = _args(resid=True, formula="y ~ x", cols=None, nointercept=True)
        result = residcols(regression_df.copy(), args)
        assert result["y_scaled"].mean() == pytest.approx(0.0, abs=0.5)

    def test_resid_original_cols_preserved(self, regression_df):
        args = _args(resid=True, formula="y ~ x", cols=None)
        result = residcols(regression_df.copy(), args)
        assert "x" in result.columns
        assert "y" in result.columns


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


class TestGrouped:
    def test_grouped_zscore_each_group_zero_mean(self, grouped_df):
        """Each group should be independently z-scored."""
        args = _args(groupcol=["group"])
        parts = []
        for _, gdf in grouped_df.groupby("group"):
            parts.append(scalecols(gdf.copy(), args))
        result = pd.concat(parts, ignore_index=True)

        for grp in ["A", "B"]:
            grp_vals = result[result["group"] == grp]["x_scaled"]
            assert grp_vals.mean() == pytest.approx(0.0, abs=1e-10)

    def test_grouped_rank_each_group_independent(self, grouped_df):
        """Rank within each group: both groups should have scores in [0, 1]."""
        args = _args(rank=True, rankdist="uniform", groupcol=["group"])
        parts = []
        for _, gdf in grouped_df.groupby("group"):
            parts.append(rankcols(gdf.copy(), args))
        result = pd.concat(parts, ignore_index=True)
        assert result["x_scaled"].min() >= 0.0
        assert result["x_scaled"].max() <= 1.0


# ---------------------------------------------------------------------------
# ScaleCommand validation
# ---------------------------------------------------------------------------


class TestScaleCommandValidation:
    def _make_file(self, df):
        import tempfile

        f = tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False)
        df.to_csv(f, sep="\t", index=False)
        f.close()
        return f.name

    def _base_args(self, fname, **kwargs):
        base = argparse.Namespace(
            DATAFILE=fname,
            cols=["x"],
            groupcol=None,
            rank=False,
            resid=False,
            shift="mean",
            scale_by="std",
            rankdist="normal",
            formula=None,
            nointercept=False,
            verbose=False,
            select=None,
            drop=None,
            move=None,
            na_rep=None,
            dropna=False,
            postquery=[],
            cast=None,
            sortasc=None,
            sortdesc=None,
            sort=None,
            round=None,
            deduplicate=None,
            noheader=False,
            removeheader=False,
            output=None,
            digits=None,
            errortag="-",
        )
        for k, v in kwargs.items():
            setattr(base, k, v)
        return base

    def test_resid_without_formula_raises(self, simple_df):
        import os

        fname = self._make_file(simple_df)
        try:
            args = self._base_args(fname, resid=True, formula=None)
            with pytest.raises(ValueError, match="formula"):
                ScaleCommand().execute(args)
        finally:
            os.unlink(fname)

    def test_missing_col_raises(self, simple_df):
        import os

        fname = self._make_file(simple_df)
        try:
            args = self._base_args(fname, cols=["no_such_col"])
            with pytest.raises(ValueError, match="no_such_col"):
                ScaleCommand().execute(args)
        finally:
            os.unlink(fname)

    def test_missing_groupcol_raises(self, simple_df):
        import os

        fname = self._make_file(simple_df)
        try:
            args = self._base_args(fname, groupcol=["no_such_group"])
            with pytest.raises(ValueError, match="no_such_group"):
                ScaleCommand().execute(args)
        finally:
            os.unlink(fname)

    def test_rank_without_cols_raises(self, simple_df):
        import os

        fname = self._make_file(simple_df)
        try:
            args = self._base_args(fname, rank=True, cols=None)
            with pytest.raises(ValueError, match="-c"):
                ScaleCommand().execute(args)
        finally:
            os.unlink(fname)
