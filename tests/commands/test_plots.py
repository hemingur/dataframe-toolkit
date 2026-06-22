"""
Tests for dfstat scat, line, hist commands.

All tests use matplotlib Agg backend and save to a temp file to avoid
any display requirements.  Tests verify:
  - The command runs without error
  - The output file is created and non-empty
  - Column validation raises ValueError for missing columns
"""

import argparse
import os
import tempfile

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from stattools.commands.hist_cmd import HistCommand
from stattools.commands.line_cmd import LineCommand
from stattools.commands.scat_cmd import ScatCommand

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    rng = np.random.default_rng(0)
    n = 30
    return pd.DataFrame(
        {
            "x": np.linspace(0, 10, n),
            "y": 2 * np.linspace(0, 10, n) + rng.normal(0, 0.5, n),
            "err": rng.uniform(0.1, 0.5, n),
            "size": rng.uniform(10, 100, n),
            "score": rng.uniform(0, 1, n),
            "group": ["A"] * 15 + ["B"] * 15,
            "cond": (["X"] * 10 + ["Y"] * 10 + ["Z"] * 10),
        }
    )


def _tsv(df):
    """Write df to a temp TSV and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False)
    df.to_csv(f, sep="\t", index=False)
    f.close()
    return f.name


def _png():
    """Return a temp PNG path (not yet created)."""
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    f.close()
    os.unlink(f.name)
    return f.name


def _base_read_args():
    """Minimal parser_read defaults needed by io.read()."""
    return dict(
        backend="pandas",
        noheader=False,
        nrows=None,
        delimiter=None,
        readasobject=None,
        prequery=[],
    )


def _scat_args(fname, outfile, **kwargs):
    base = dict(
        DATAFILE=fname,
        xcol="x",
        ycol="y",
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        logx=False,
        logy=False,
        xmargin=False,
        ymargin=False,
        xticks=None,
        yticks=None,
        title="",
        size="5x3.5",
        fontsize="screen",
        styles=None,
        usetex=False,
        file=outfile,
        groupcol=None,
        subgraphcol=None,
        ncols=None,
        legend=None,
        legendtitle=None,
        legendloc=None,
        marker="o",
        sizecol=None,
        colorcol=None,
        fit=False,
        robust=False,
        nointercept=False,
        pvalue=False,
        **_base_read_args(),
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _line_args(fname, outfile, **kwargs):
    base = dict(
        DATAFILE=fname,
        xcol="x",
        ycol="y",
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        logx=False,
        logy=False,
        xmargin=False,
        ymargin=False,
        xticks=None,
        yticks=None,
        title="",
        size="5x3.5",
        fontsize="screen",
        styles=None,
        usetex=False,
        file=outfile,
        groupcol=None,
        subgraphcol=None,
        ncols=None,
        legend=None,
        legendtitle=None,
        legendloc=None,
        marker="",
        drawstyle="default",
        yerr=None,
        yci=None,
        fit=False,
        robust=False,
        weights=None,
        nointercept=False,
        pvalue=False,
        **_base_read_args(),
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _hist_args(fname, outfile, **kwargs):
    base = dict(
        DATAFILE=fname,
        xcol="x",
        weightcol=None,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        logx=False,
        logy=False,
        xmargin=False,
        ymargin=False,
        title="",
        size="5x3.5",
        fontsize="screen",
        styles=None,
        usetex=False,
        file=outfile,
        groupcol=None,
        subgraphcol=None,
        ncols=None,
        legend=None,
        legendtitle=None,
        legendloc=None,
        normed=False,
        cumulative=False,
        kde=False,
        stats=False,
        bins=10,
        binwidth=None,
        **_base_read_args(),
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def _assert_file_created(path):
    assert os.path.exists(path), f"Output file not created: {path}"
    assert os.path.getsize(path) > 0, f"Output file is empty: {path}"


# ---------------------------------------------------------------------------
# ScatCommand
# ---------------------------------------------------------------------------


class TestScatCommand:
    def test_basic_scatter(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(_scat_args(fname, out))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, groupcol=["group"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, subgraphcol=["cond"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol_and_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(
                _scat_args(fname, out, subgraphcol=["cond"], groupcol=["group"])
            )
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_sizecol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, sizecol="size"))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_colorcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, colorcol="score"))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_publication_fontsize(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ScatCommand().execute(
                _scat_args(fname, out, fontsize="publication", size="single")
            )
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_missing_xcol_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError, match="no_such"):
                ScatCommand().execute(_scat_args(fname, out, xcol="no_such"))
        finally:
            os.unlink(fname)

    def test_missing_xcol_arg_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError):
                ScatCommand().execute(_scat_args(fname, out, xcol=None))
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# LineCommand
# ---------------------------------------------------------------------------


class TestLineCommand:
    def test_basic_line(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            LineCommand().execute(_line_args(fname, out))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            LineCommand().execute(_line_args(fname, out, groupcol=["group"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_yerr(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            LineCommand().execute(_line_args(fname, out, yerr="err"))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            LineCommand().execute(_line_args(fname, out, subgraphcol=["cond"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_missing_ycol_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError):
                LineCommand().execute(_line_args(fname, out, ycol=None))
        finally:
            os.unlink(fname)

    def test_missing_yerr_col_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError, match="no_err"):
                LineCommand().execute(_line_args(fname, out, yerr="no_err"))
        finally:
            os.unlink(fname)


# ---------------------------------------------------------------------------
# HistCommand
# ---------------------------------------------------------------------------


class TestHistCommand:
    def test_basic_hist(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_normed(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, normed=True))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_cumulative(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, cumulative=True))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_kde(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, kde=True))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, groupcol=["group"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_groupcol_kde(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, groupcol=["group"], kde=True))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, subgraphcol=["cond"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_stats_flag(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, stats=True))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_binwidth(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            HistCommand().execute(_hist_args(fname, out, binwidth=1.0, bins=10))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_missing_xcol_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError, match="no_col"):
                HistCommand().execute(_hist_args(fname, out, xcol="no_col"))
        finally:
            os.unlink(fname)
