"""Tests for stattools.commands.dataset_cmd."""

import argparse
import io as _io
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stattools.commands.dataset_cmd import (
    DatasetCommand,
    _load_seaborn,
    _load_statsmodels,
    _list_seaborn,
    _list_statsmodels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal iris-like DataFrame for mocking
_IRIS_DF = pd.DataFrame({
    "sepal_length": [5.1, 4.9, 4.7],
    "sepal_width":  [3.5, 3.0, 3.2],
    "species":      ["setosa", "setosa", "versicolor"],
})

_TITANIC_DF = pd.DataFrame({
    "survived": [0, 1, 1],
    "pclass":   [3, 1, 3],
    "sex":      ["male", "female", "female"],
})


def _make_args(**kwargs):
    defaults = dict(
        NAME=None,
        source=None,
        list_datasets=False,
        output=None,
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
        expect=None,
        round=None,
        sigdig=None,
        movetofront=None,
        movetoback=None,
        deduplicate=None,
        noheader=False,
        removeheader=False,
        digits=None,
        errortag="-",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _capture_execute(args):
    """Run DatasetCommand.execute() and return stdout as list-of-dicts (TSV rows)."""
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        DatasetCommand().execute(args)
    finally:
        sys.stdout = old
    lines = [l for l in buf.getvalue().splitlines() if l.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    return [dict(zip(header, l.split("\t"))) for l in lines[1:]]


def _capture_raw(args):
    """Run DatasetCommand.execute() and return raw stdout string."""
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        DatasetCommand().execute(args)
    finally:
        sys.stdout = old
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests: load helpers (mocked)
# ---------------------------------------------------------------------------

class TestLoadHelpers:

    def test_load_seaborn_calls_sns(self):
        with patch("seaborn.load_dataset", return_value=_IRIS_DF) as mock_load:
            df = _load_seaborn("iris")
            mock_load.assert_called_once_with("iris")
            assert len(df) == 3

    def test_load_statsmodels_known_dataset(self):
        # longley is bundled with statsmodels — no network needed
        df = _load_statsmodels("longley")
        assert "TOTEMP" in df.columns
        assert len(df) == 16

    def test_load_statsmodels_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            _load_statsmodels("__no_such_dataset__")

    def test_list_seaborn_returns_triples(self):
        with patch("seaborn.get_dataset_names", return_value=["iris", "tips"]):
            rows = _list_seaborn()
        assert all(len(r) == 3 for r in rows)
        assert all(r[0] == "seaborn" for r in rows)
        names = [r[1] for r in rows]
        assert "iris" in names
        assert "tips" in names

    def test_list_statsmodels_returns_triples(self):
        rows = _list_statsmodels()
        assert len(rows) > 0
        assert all(len(r) == 3 for r in rows)
        assert all(r[0] == "statsmodels" for r in rows)
        # longley is always present in statsmodels and has a TITLE
        by_name = {r[1]: r[2] for r in rows}
        assert "longley" in by_name
        assert by_name["longley"] != ""


# ---------------------------------------------------------------------------
# Unit tests: DatasetCommand.execute() (mocked)
# ---------------------------------------------------------------------------

class TestDatasetCommandMocked:

    def test_no_name_no_list_raises(self):
        args = _make_args(NAME=None, list_datasets=False)
        with pytest.raises(ValueError, match="--list"):
            DatasetCommand().execute(args)

    def test_unknown_dataset_raises(self):
        args = _make_args(NAME="__no_such_dataset__", source=["seaborn"])
        with patch("seaborn.load_dataset", side_effect=ValueError("not found")):
            with pytest.raises(ValueError, match="not found"):
                DatasetCommand().execute(args)

    def test_load_returns_tsv_rows(self):
        with patch("seaborn.load_dataset", return_value=_IRIS_DF):
            args = _make_args(NAME="iris", source=["seaborn"])
            rows = _capture_execute(args)
        assert len(rows) == 3
        assert "sepal_length" in rows[0]
        assert "species" in rows[0]

    def test_source_filter_seaborn_only(self):
        with patch("seaborn.load_dataset", return_value=_IRIS_DF) as mock_sns:
            args = _make_args(NAME="iris", source=["seaborn"])
            _capture_execute(args)
        mock_sns.assert_called_once_with("iris")

    def test_source_filter_statsmodels_not_tried_when_seaborn_specified(self):
        with patch("seaborn.load_dataset", return_value=_IRIS_DF):
            with patch("statsmodels.api.datasets") as mock_sm:
                args = _make_args(NAME="iris", source=["seaborn"])
                _capture_execute(args)
        mock_sm.assert_not_called()

    def test_falls_through_to_second_source(self):
        """If seaborn fails, statsmodels is tried next."""
        with patch("seaborn.load_dataset", side_effect=ValueError("not in seaborn")):
            args = _make_args(NAME="longley", source=["seaborn", "statsmodels"])
            rows = _capture_execute(args)
        # longley has TOTEMP column
        assert "TOTEMP" in rows[0]

    def test_all_sources_fail_raises(self):
        with patch("seaborn.load_dataset", side_effect=ValueError("no")):
            args = _make_args(NAME="__no_such__", source=["seaborn"])
            with pytest.raises(ValueError, match="not found"):
                DatasetCommand().execute(args)

    def test_list_prints_header(self):
        with patch("seaborn.get_dataset_names", return_value=["iris"]):
            args = _make_args(list_datasets=True, source=["seaborn"])
            raw = _capture_raw(args)
        assert raw.startswith("source\tname\tdescription")

    def test_list_seaborn_rows(self):
        with patch("seaborn.get_dataset_names", return_value=["iris", "tips"]):
            args = _make_args(list_datasets=True, source=["seaborn"])
            raw = _capture_raw(args)
        lines = raw.strip().splitlines()
        names = [l.split("\t")[1] for l in lines[1:]]
        assert "iris" in names
        assert "tips" in names

    def test_list_source_column_correct(self):
        with patch("seaborn.get_dataset_names", return_value=["iris"]):
            args = _make_args(list_datasets=True, source=["seaborn"])
            raw = _capture_raw(args)
        data_line = raw.strip().splitlines()[1]
        source_val = data_line.split("\t")[0]
        assert source_val == "seaborn"

    def test_list_does_not_call_load(self):
        with patch("seaborn.load_dataset") as mock_load:
            with patch("seaborn.get_dataset_names", return_value=["iris"]):
                args = _make_args(list_datasets=True, source=["seaborn"])
                _capture_raw(args)
        mock_load.assert_not_called()

    def test_default_source_tries_all(self):
        """When source is None, all registered sources are tried."""
        with patch("seaborn.load_dataset", return_value=_IRIS_DF) as mock_sns:
            args = _make_args(NAME="iris", source=None)
            _capture_execute(args)
        mock_sns.assert_called_once()

    def test_row_count_matches_dataset(self):
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        with patch("seaborn.load_dataset", return_value=df):
            args = _make_args(NAME="any", source=["seaborn"])
            rows = _capture_execute(args)
        assert len(rows) == 10


# ---------------------------------------------------------------------------
# Integration tests: real bundled datasets (no network needed)
# ---------------------------------------------------------------------------

class TestDatasetIntegration:

    def test_seaborn_iris_loads(self):
        """seaborn iris is cached locally."""
        args = _make_args(NAME="iris", source=["seaborn"])
        rows = _capture_execute(args)
        assert len(rows) == 150
        assert "sepal_length" in rows[0]
        assert "species" in rows[0]

    def test_statsmodels_longley_loads(self):
        """statsmodels longley is bundled — no network needed."""
        args = _make_args(NAME="longley", source=["statsmodels"])
        rows = _capture_execute(args)
        assert len(rows) == 16
        assert "TOTEMP" in rows[0]

    def test_list_seaborn_integration(self):
        args = _make_args(list_datasets=True, source=["seaborn"])
        raw = _capture_raw(args)
        lines = raw.strip().splitlines()
        assert lines[0] == "source\tname\tdescription"
        names = {l.split("\t")[1] for l in lines[1:]}
        assert "iris" in names

    def test_list_statsmodels_integration(self):
        args = _make_args(list_datasets=True, source=["statsmodels"])
        raw = _capture_raw(args)
        names = {l.split("\t")[1] for l in raw.strip().splitlines()[1:]}
        assert "longley" in names

    def test_list_all_sources_no_duplicate_headers(self):
        args = _make_args(list_datasets=True, source=["seaborn", "statsmodels"])
        raw = _capture_raw(args)
        header_count = sum(1 for l in raw.splitlines() if l.startswith("source\t"))
        assert header_count == 1

    def test_fallthrough_seaborn_to_statsmodels(self):
        """longley is not in seaborn; fallthrough should find it in statsmodels."""
        args = _make_args(NAME="longley", source=["seaborn", "statsmodels"])
        rows = _capture_execute(args)
        assert len(rows) == 16
