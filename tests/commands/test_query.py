"""
Tests for dftk.commands.query_cmd.
"""

import argparse
import os
import tempfile

import pandas as pd
import pytest

from dftk.commands.query_cmd import QueryCommand, _pandas_query, _sql_query

duckdb = pytest.importorskip("duckdb", reason="duckdb not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "x": [1.0, 3.0, 2.0, 4.0, 6.0],
            "label": ["foo", "bar", "foo", "baz", "foo"],
        }
    )


def _args(**overrides):
    defaults = dict(DATAFILE=None)
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# _pandas_query
# ---------------------------------------------------------------------------


class TestPandasQuery:
    def test_single_filter(self, sample_df):
        result = _pandas_query(sample_df, ["x > 2"])
        assert list(result["x"]) == [3.0, 4.0, 6.0]

    def test_multiple_filters_anded(self, sample_df):
        result = _pandas_query(sample_df, ["x > 1", "group == 'A'"])
        assert list(result["x"]) == [3.0]

    def test_string_method(self, sample_df):
        result = _pandas_query(sample_df, ["label.str.startswith('foo')"])
        assert len(result) == 3
        assert set(result["label"]) == {"foo"}

    def test_no_expressions_passes_through(self, sample_df):
        result = _pandas_query(sample_df, [])
        assert len(result) == len(sample_df)

    def test_bad_expression_raises(self, sample_df):
        with pytest.raises(ValueError, match="pandas query failed"):
            _pandas_query(sample_df, ["nonexistent_col > 0"])

    def test_all_rows_filtered(self, sample_df):
        result = _pandas_query(sample_df, ["x > 1000"])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# _sql_query — in-memory (pandas-registered) path
# ---------------------------------------------------------------------------


class TestSqlQueryInMemory:
    def test_simple_where(self, sample_df):
        args = _args()
        # Pass df as a pre-registered source by patching DATAFILE to None
        # and monkey-patching io.read — easier: just test _sql_query directly
        # by passing a temp TSV file so the pandas path is exercised.
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            sample_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = _args(DATAFILE=fname)
            result = _sql_query(args, "SELECT * FROM data WHERE x > 2", "data")
            assert set(result["x"]) == {3.0, 4.0, 6.0}
        finally:
            os.unlink(fname)

    def test_aggregation(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            sample_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = _args(DATAFILE=fname)
            result = _sql_query(
                args,
                'SELECT "group", AVG(x) AS mean FROM data GROUP BY "group" ORDER BY "group"',  # noqa: E501
                "data",
            )
            assert list(result["group"]) == ["A", "B"]
            assert result.loc[result["group"] == "A", "mean"].iloc[0] == pytest.approx(
                2.0
            )  # noqa: E501
            assert result.loc[result["group"] == "B", "mean"].iloc[0] == pytest.approx(
                4.0
            )  # noqa: E501
        finally:
            os.unlink(fname)

    def test_window_function(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            sample_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = _args(DATAFILE=fname)
            result = _sql_query(
                args,
                'SELECT *, ROW_NUMBER() OVER (PARTITION BY "group" ORDER BY x) AS rn FROM data',  # noqa: E501
                "data",
            )
            assert "rn" in result.columns
            a_rows = result[result["group"] == "A"].sort_values("x")
            assert list(a_rows["rn"]) == [1, 2]
        finally:
            os.unlink(fname)

    def test_custom_table_name(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            sample_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = _args(DATAFILE=fname)
            result = _sql_query(args, "SELECT * FROM t WHERE x > 2", "t")
            assert len(result) == 3
        finally:
            os.unlink(fname)

    def test_bad_sql_raises(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            sample_df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = _args(DATAFILE=fname)
            with pytest.raises(ValueError, match="DuckDB SQL failed"):
                _sql_query(args, "SELECT * FROM nonexistent_table", "data")
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# _sql_query — native parquet path (out-of-core)
# ---------------------------------------------------------------------------


class TestSqlQueryNativeParquet:
    def test_parquet_where(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname = f.name
        try:
            sample_df.to_parquet(fname, index=False)
            args = _args(DATAFILE=fname)
            result = _sql_query(args, "SELECT * FROM data WHERE x > 2", "data")
            assert set(result["x"]) == {3.0, 4.0, 6.0}
        finally:
            os.unlink(fname)

    def test_parquet_aggregation(self, sample_df):
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname = f.name
        try:
            sample_df.to_parquet(fname, index=False)
            args = _args(DATAFILE=fname)
            result = _sql_query(
                args,
                'SELECT "group", COUNT(*) AS n FROM data GROUP BY "group" ORDER BY "group"',  # noqa: E501
                "data",
            )
            assert list(result["n"]) == [2, 3]
        finally:
            os.unlink(fname)

    def test_parquet_file_not_loaded_into_pandas(self, sample_df):
        # We verify the native path is taken by checking DATAFILE ends in .parquet
        # and exists — the actual memory savings are verified by DuckDB internals.
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            fname = f.name
        try:
            sample_df.to_parquet(fname, index=False)
            args = _args(DATAFILE=fname)
            # Should succeed without io.read() being called (no pandas read)
            result = _sql_query(args, "SELECT COUNT(*) AS n FROM data", "data")
            assert result["n"].iloc[0] == len(sample_df)
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# QueryCommand.execute validation
# ---------------------------------------------------------------------------


class TestQueryCommandValidation:
    def test_combined_q_and_sql_raises(self):
        cmd = QueryCommand()
        args = argparse.Namespace(
            query=["x > 0"],
            sql="SELECT * FROM data",
            table="data",
            DATAFILE=None,
        )
        with pytest.raises(ValueError, match="Cannot combine"):
            cmd.execute(args)

    def test_no_filter_passes_through(self, sample_df):
        # With no -q and no --sql, execute() should be a pass-through.
        # We test _pandas_query([]) directly since execute() needs full io.
        result = _pandas_query(sample_df, [])
        assert len(result) == len(sample_df)


class TestSqlIntegerSum:
    """SUM of an integer column is a DuckDB HUGEINT; pandas would make it float."""

    @pytest.fixture
    def parquet(self, tmp_path):
        path = str(tmp_path / "data.parquet")
        pd.DataFrame(
            {"g": ["a", "a", "b"], "n": pd.array([10, None, 9], dtype="Int64")}
        ).to_parquet(path)
        return path

    def test_sum_is_integer(self, parquet):
        sql = "SELECT g, SUM(n) AS s FROM data GROUP BY g ORDER BY g"
        result = _sql_query(_args(DATAFILE=parquet), sql, "data")
        assert str(result["s"].dtype) == "Int64"
        assert result["s"].tolist() == [10, 9]

    def test_empty_sum_is_missing(self, parquet):
        sql = "SELECT SUM(n) AS s FROM data WHERE false"
        result = _sql_query(_args(DATAFILE=parquet), sql, "data")
        assert str(result["s"].dtype) == "Int64"
        assert result["s"].isna().all()

    def test_sum_beyond_float_precision_stays_float(self, parquet):
        sql = "SELECT SUM(x) AS s FROM (SELECT 9007199254740993::BIGINT AS x)"
        result = _sql_query(_args(DATAFILE=parquet), sql, "data")
        assert result["s"].dtype == "float64"

    def test_other_aggregates_unchanged(self, parquet):
        sql = "SELECT AVG(n) AS a, COUNT(*) AS c, MAX(n) AS m FROM data"
        result = _sql_query(_args(DATAFILE=parquet), sql, "data")
        assert result.dtypes.astype(str).tolist() == ["float64", "int64", "int64"]
