"""
Shared pytest fixtures for stattools tests.

All datasets are synthetic — constructed from numpy/pandas directly so that
expected statistical values are exactly known without extra dependencies.
"""

import argparse

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Five rows, two numeric columns, no groups.

    x: [1, 2, 3, 4, 5]  — mean=3, symmetric (skew=0)
    y: [10, 20, 30, 40, 50]  — y == 10*x throughout
    """
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )


@pytest.fixture
def grouped_df() -> pd.DataFrame:
    """Two groups (A, B), three rows each.

    group A: value = [1, 2, 3]   — mean=2, sum=6
    group B: value = [10, 20, 30] — mean=20, sum=60
    """
    return pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def two_group_two_col_df() -> pd.DataFrame:
    """Two groups, two numeric columns — tests multi-column grouped output."""
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "x": [1.0, 3.0, 10.0, 30.0],
            "y": [2.0, 4.0, 20.0, 40.0],
        }
    )


# ---------------------------------------------------------------------------
# Args namespace factory
# ---------------------------------------------------------------------------


def make_args(**overrides) -> argparse.Namespace:
    """Return an args Namespace with stat-command defaults.

    Pass keyword arguments to override any field.  For example::

        args = make_args(cols=["x"], groupcol=["group"])
    """
    defaults = dict(
        cols=["x"],
        groupcol=None,
        confidencelevel=95.0,
        confidencemethod="linear",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def args_factory():
    """Fixture exposing make_args so tests can call it with custom kwargs."""
    return make_args
