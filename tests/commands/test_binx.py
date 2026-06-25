"""Tests for stattools.commands.binx_cmd._binx and _parse_binspec."""

import numpy as np
import pandas as pd
import pytest

from stattools.commands.binx_cmd import _binx, _parse_binspec
from tests.conftest import make_args


def _args(**kwargs):
    defaults = dict(col="x", destcol=None, binspec="0:5:1", usevalue=None)
    defaults.update(kwargs)
    return make_args(**defaults)


# ---------------------------------------------------------------------------
# _parse_binspec
# ---------------------------------------------------------------------------


class TestParseBinspec:
    def test_range_spec(self):
        edges = _parse_binspec("0:5:1")
        assert edges == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_range_spec_float_step(self):
        edges = _parse_binspec("0:1:0.5")
        assert edges == pytest.approx([0.0, 0.5])

    def test_comma_separated(self):
        edges = _parse_binspec("0,5,10,50")
        assert edges == [0.0, 5.0, 10.0, 50.0]

    def test_negative_edges(self):
        edges = _parse_binspec("-3:4:1")
        assert edges == pytest.approx([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

    def test_bad_range_spec_raises(self):
        with pytest.raises(ValueError, match="binspec must be"):
            _parse_binspec("0:5")

    def test_too_few_edges_raises(self):
        # arange produces only one edge → _binx should raise
        df = pd.DataFrame({"x": [1.0]})
        with pytest.raises(ValueError, match="at least 2 edges"):
            _binx(df, _args(binspec="0:1:5"))


# ---------------------------------------------------------------------------
# Basic binning
# ---------------------------------------------------------------------------


class TestBinxBasic:
    def test_default_destcol_name(self):
        df = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
        result = _binx(df, _args(binspec="0:4:1"))
        assert "x_bin" in result.columns

    def test_custom_destcol(self):
        df = pd.DataFrame({"x": [0.5]})
        result = _binx(df, _args(destcol="bucket", binspec="0:4:1"))
        assert "bucket" in result.columns

    def test_bin_indices_correct(self):
        df = pd.DataFrame({"x": [0.5, 1.5, 2.5]})
        result = _binx(df, _args(binspec="0:4:1"))
        assert list(result["x_bin"]) == pytest.approx([0.0, 1.0, 2.0])

    def test_include_lowest(self):
        # value exactly at the lower boundary should land in bin 0
        df = pd.DataFrame({"x": [0.0]})
        result = _binx(df, _args(binspec="0:3:1"))
        assert result["x_bin"].iloc[0] == pytest.approx(0.0)

    def test_out_of_range_is_nan(self):
        df = pd.DataFrame({"x": [100.0]})
        result = _binx(df, _args(binspec="0:5:1"))
        assert pd.isna(result["x_bin"].iloc[0])

    def test_original_columns_preserved(self):
        df = pd.DataFrame({"x": [1.0], "y": [99.0]})
        result = _binx(df, _args(binspec="0:5:1"))
        assert "y" in result.columns
        assert result["y"].iloc[0] == 99.0

    def test_comma_spec_bins(self):
        df = pd.DataFrame({"x": [1.0, 6.0, 20.0]})
        result = _binx(df, _args(binspec="0,5,10,50"))
        assert list(result["x_bin"]) == pytest.approx([0.0, 1.0, 2.0])


# ---------------------------------------------------------------------------
# usevalue replacement
# ---------------------------------------------------------------------------


class TestBinxUseValue:
    @pytest.fixture
    def df(self):
        # bins: [0,1), [1,2), [2,3)  →  indices 0, 1, 2
        return pd.DataFrame({"x": [0.5, 1.5, 2.5]})

    def test_usevalue_lower(self, df):
        result = _binx(df, _args(binspec="0:4:1", usevalue="l"))
        assert list(result["x_bin"]) == pytest.approx([0.0, 1.0, 2.0])

    def test_usevalue_upper(self, df):
        result = _binx(df, _args(binspec="0:4:1", usevalue="u"))
        assert list(result["x_bin"]) == pytest.approx([1.0, 2.0, 3.0])

    def test_usevalue_mid(self, df):
        result = _binx(df, _args(binspec="0:4:1", usevalue="m"))
        assert list(result["x_bin"]) == pytest.approx([0.5, 1.5, 2.5])

    def test_usevalue_nan_propagates(self):
        df = pd.DataFrame({"x": [0.5, 999.0]})
        result = _binx(df, _args(binspec="0:3:1", usevalue="m"))
        assert result["x_bin"].iloc[0] == pytest.approx(0.5)
        assert pd.isna(result["x_bin"].iloc[1])

    def test_usevalue_unequal_bins_mid(self):
        # edges 0, 5, 10, 50 → midpoints 2.5, 7.5, 30.0
        df = pd.DataFrame({"x": [1.0, 6.0, 20.0]})
        result = _binx(df, _args(binspec="0,5,10,50", usevalue="m"))
        assert list(result["x_bin"]) == pytest.approx([2.5, 7.5, 30.0])


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBinxEdgeCases:
    def test_all_nan_input(self):
        df = pd.DataFrame({"x": [float("nan"), float("nan")]})
        result = _binx(df, _args(binspec="0:5:1"))
        assert result["x_bin"].isna().all()

    def test_single_value_dataframe(self):
        # pd.cut uses right-closed intervals (a, b], so 2.0 lands in (1, 2] → index 1
        df = pd.DataFrame({"x": [2.0]})
        result = _binx(df, _args(binspec="0:5:1"))
        assert result["x_bin"].iloc[0] == pytest.approx(1.0)

    def test_many_rows(self):
        rng = np.random.default_rng(42)
        vals = rng.uniform(0, 10, 1000)
        df = pd.DataFrame({"x": vals})
        result = _binx(df, _args(binspec="0:11:1"))
        valid = result["x_bin"].dropna()
        assert (valid >= 0).all() and (valid <= 9).all()
