"""
Tests for dftk scat, line, hist commands.

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

from dftk.commands.chi_cmd import ChiCommand
from dftk.commands.hist_cmd import HistCommand
from dftk.commands.line_cmd import LineCommand
from dftk.commands.scat_cmd import ScatCommand

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
        fit=None,
        frac=2 / 3,
        robust=False,
        nointercept=False,
        pvalue=False,
        hexbin=False,
        gridsize=100,
        hexlog=False,
        xerr=None,
        yerr=None,
        errscale=1.0,
        weights=None,
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


def _chi_args(fname, outfile, **kwargs):
    base = dict(
        DATAFILE=fname,
        xcol="x",
        ycol="y",
        xlabel=None,
        ylabel=None,
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


@pytest.fixture
def cloud_df():
    """400 points, y = 2x + noise, in two groups with different slopes."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 10, 400)
    group = np.where(np.arange(400) % 2 == 0, "a", "b")
    slope = np.where(group == "a", 2.0, -1.0)
    y = slope * x + rng.normal(0, 0.5, 400)
    return pd.DataFrame({"x": x, "y": y, "group": group, "w": x * 10})


def _draw(df, **kwargs):
    """Draw one scat panel and return its Axes (no file written)."""
    import matplotlib.pyplot as plt

    from dftk.commands.scat_cmd import _scatter_ax

    fig, ax = plt.subplots()
    _scatter_ax(ax, df, _scat_args(None, None, **kwargs))
    return ax


def _hexbins(ax):
    from matplotlib.collections import PolyCollection

    return [c for c in ax.collections if isinstance(c, PolyCollection)]


class TestScatHexbin:
    def test_counts_add_up_to_points(self, cloud_df):
        ax = _draw(cloud_df, hexbin=True, gridsize=10)
        (hb,) = _hexbins(ax)
        assert hb.get_array().sum() == len(cloud_df)
        assert hb.get_array().min() >= 1  # empty hexagons are not drawn

    def test_colorcol_gives_mean_per_hexagon(self, cloud_df):
        ax = _draw(cloud_df, hexbin=True, gridsize=10, colorcol="w")
        (hb,) = _hexbins(ax)
        # w = 10 x, so every hexagon's mean lies within the range of w
        assert cloud_df["w"].min() <= hb.get_array().min()
        assert hb.get_array().max() <= cloud_df["w"].max()
        assert ax.figure.axes[-1].get_ylabel() == "mean w"

    def test_log_axes_bin_on_log_scale(self, cloud_df):
        df = cloud_df.assign(y=cloud_df["x"] ** 2)
        ax = _draw(df, hexbin=True, logx=True, logy=True)
        assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
        (hb,) = _hexbins(ax)
        assert hb.get_array().sum() == len(df)

    def test_log_axes_drop_non_positive(self, cloud_df, capsys):
        ax = _draw(cloud_df, hexbin=True, logy=True)
        (hb,) = _hexbins(ax)
        n_positive = (cloud_df["y"] > 0).sum()
        assert hb.get_array().sum() == n_positive
        assert f"dropped {len(cloud_df) - n_positive} rows" in capsys.readouterr().err

    def test_hexlog_uses_log_norm(self, cloud_df):
        from matplotlib.colors import LogNorm

        (hb,) = _hexbins(_draw(cloud_df, hexbin=True, hexlog=True))
        assert isinstance(hb.norm, LogNorm)

    def test_subgraphs_share_colour_scale(self, cloud_df):
        import matplotlib.pyplot as plt

        from dftk.commands.scat_cmd import _scatter_ax, _share_hexbin_colours

        fig, axes = plt.subplots(1, 2, squeeze=False)
        big = cloud_df
        small = cloud_df.head(40)
        args = _scat_args(None, None, hexbin=True, gridsize=5)
        _scatter_ax(axes[0][0], big, args)
        _scatter_ax(axes[0][1], small, args)
        _share_hexbin_colours(axes)
        (a,) = _hexbins(axes[0][0])
        (b,) = _hexbins(axes[0][1])
        assert a.get_clim() == b.get_clim()
        assert a.get_clim()[1] == a.get_array().max()

    @pytest.mark.parametrize("option", [{"groupcol": ["group"]}, {"sizecol": "w"}])
    def test_rejects_groupcol_and_sizecol(self, cloud_df, option):
        fname = _tsv(cloud_df)
        try:
            with pytest.raises(ValueError, match="--hexbin cannot be combined"):
                ScatCommand().execute(_scat_args(fname, _png(), hexbin=True, **option))
        finally:
            os.unlink(fname)

    def test_writes_file_with_fit(self, cloud_df):
        fname, out = _tsv(cloud_df), _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, hexbin=True, fit="lowess"))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)


class TestScatFit:
    def test_bare_fit_is_ols(self):
        from dftk.commands.scat_cmd import ScatCommand as Cmd

        parser = argparse.ArgumentParser()
        Cmd().add_arguments(parser)
        assert parser.parse_args(["-", "-x", "a", "-y", "b", "--fit"]).fit == "ols"
        assert parser.parse_args(["-", "-x", "a", "-y", "b"]).fit is None
        args = parser.parse_args(["-", "-x", "a", "-y", "b", "--fit", "lowess"])
        assert args.fit == "lowess"

    def test_ols_line_has_fitted_slope(self, cloud_df, capsys):
        df = cloud_df[cloud_df["group"] == "a"]
        (line,) = _draw(df, fit="ols").get_lines()
        x, y = line.get_xdata(), line.get_ydata()
        assert (y[-1] - y[0]) / (x[-1] - x[0]) == pytest.approx(2.0, abs=0.05)

    def test_one_line_per_group_in_group_colour(self, cloud_df, capsys):
        ax = _draw(cloud_df, fit="ols", groupcol=["group"])
        lines = ax.get_lines()
        assert len(lines) == 2
        slopes = sorted(
            (ln.get_ydata()[-1] - ln.get_ydata()[0])
            / (ln.get_xdata()[-1] - ln.get_xdata()[0])
            for ln in lines
        )
        assert slopes == pytest.approx([-1.0, 2.0], abs=0.05)
        points = [c for c in ax.collections if c not in _hexbins(ax)]
        point_colours = {tuple(c.get_facecolor()[0]) for c in points}
        assert {tuple(matplotlib.colors.to_rgba(ln.get_color())) for ln in lines} == (
            point_colours
        )
        err = capsys.readouterr().err
        assert "== a" in err and "== b" in err  # a summary per group

    def test_group_line_spans_group_x_range(self, cloud_df, capsys):
        df = cloud_df.copy()
        df.loc[df["group"] == "b", "x"] += 20
        ax = _draw(df, fit="ols", groupcol=["group"])
        spans = sorted(
            (ln.get_xdata().min(), ln.get_xdata().max()) for ln in ax.get_lines()
        )
        for (lo, hi), (_, g) in zip(spans, df.groupby("group"), strict=True):
            assert (lo, hi) == pytest.approx((g["x"].min(), g["x"].max()))

    def test_lowess_follows_a_curve(self):
        from dftk.commands.scat_cmd import _lowess_curve

        rng = np.random.default_rng(1)
        x = rng.uniform(0, 2 * np.pi, 2000)
        y = np.sin(x) + rng.normal(0, 0.1, 2000)
        xs, ys = _lowess_curve(x, y, _scat_args(None, None, frac=0.1))
        assert np.all(np.diff(xs) >= 0)
        inner = (xs > 0.5) & (xs < 2 * np.pi - 0.5)
        assert np.abs(ys[inner] - np.sin(xs[inner])).max() < 0.1

    def test_lowess_legend(self, cloud_df):
        ax = _draw(cloud_df, fit="lowess", frac=0.3)
        assert ax.get_legend().get_texts()[0].get_text() == "LOWESS (frac=0.3)"

    def test_too_few_points_skipped(self, capsys):
        df = pd.DataFrame({"x": [1.0, 2.0, None], "y": [1.0, 2.0, 3.0]})
        ax = _draw(df, fit="ols")
        assert ax.get_lines() == []
        assert "fewer than 3 points" in capsys.readouterr().err

    @pytest.mark.parametrize("frac", [0, 1.5])
    def test_frac_out_of_range(self, cloud_df, frac):
        fname = _tsv(cloud_df)
        try:
            with pytest.raises(ValueError, match="--frac"):
                ScatCommand().execute(
                    _scat_args(fname, _png(), fit="lowess", frac=frac)
                )
        finally:
            os.unlink(fname)


def _errorbar_segments(ax):
    """{"x" or "y": [((x0, y0), (x1, y1)), ...]} of the error-bar lines."""
    from matplotlib.collections import LineCollection

    out = {"x": [], "y": []}
    for c in ax.collections:
        if isinstance(c, LineCollection):
            for (x0, y0), (x1, y1) in c.get_segments():
                out["x" if y0 == y1 else "y"].append(((x0, y0), (x1, y1)))
    return out


class TestScatErrorbars:
    def test_yerr_and_xerr_with_scale(self, simple_df):
        df = simple_df.assign(xe=0.5)
        ax = _draw(df, yerr="err", xerr="xe", errscale=2.0)
        seg = _errorbar_segments(ax)
        (x0, y0), (x1, y1) = seg["y"][0]
        assert y1 - y0 == pytest.approx(4 * df["err"].iloc[0])
        (x0, y0), (x1, y1) = seg["x"][0]
        assert x1 - x0 == pytest.approx(2.0)

    def test_group_colour(self, simple_df):
        from matplotlib.collections import LineCollection

        ax = _draw(simple_df, yerr="err", groupcol=["group"])
        bars = [c for c in ax.collections if isinstance(c, LineCollection)]
        points = [c for c in ax.collections if not isinstance(c, LineCollection)]
        assert {tuple(b.get_color()[0][:3]) for b in bars} == {
            tuple(p.get_facecolor()[0][:3]) for p in points
        }

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(yerr="neg"), "negative"),
            (dict(yerr="nosuch"), "nosuch"),
            (dict(yerr="err", errscale=0), "--errscale"),
            (dict(yerr="err", hexbin=True), "--hexbin"),
        ],
    )
    def test_errors(self, simple_df, kwargs, match):
        fname = _tsv(simple_df.assign(neg=-1.0))
        try:
            with pytest.raises(ValueError, match=match):
                ScatCommand().execute(_scat_args(fname, _png(), **kwargs))
        finally:
            os.unlink(fname)

    def test_writes_file(self, simple_df):
        fname, out = _tsv(simple_df), _png()
        try:
            ScatCommand().execute(_scat_args(fname, out, yerr="err", xerr="err"))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            os.unlink(out)


class TestScatWeightedFit:
    def _slope(self, ax):
        (line,) = ax.get_lines()
        x, y = line.get_xdata(), line.get_ydata()
        return (y[-1] - y[0]) / (x[-1] - x[0])

    def test_weights_pull_the_line(self, capsys):
        # two clusters on different lines; heavy weights on one decide the slope
        x = np.array([1.0, 2, 3, 4, 1, 2, 3, 4])
        y = np.array([1.0, 2, 3, 4, 3, 6, 9, 12])
        w = np.array([1000.0] * 4 + [0.001] * 4)
        df = pd.DataFrame({"x": x, "y": y, "w": w})
        unweighted = self._slope(_draw(df, fit="ols"))
        weighted = _draw(df, fit="ols", weights="w")
        assert self._slope(weighted) == pytest.approx(1.0, abs=1e-3)
        assert unweighted == pytest.approx(2.0, abs=0.01)
        legend = weighted.get_legend().get_texts()[0].get_text()
        assert "(weighted)" in legend

    def test_through_origin_ivw(self, capsys):
        rng = np.random.default_rng(3)
        bx = rng.uniform(0.05, 0.5, 200)
        se = 0.01 + 0.05 * bx
        df = pd.DataFrame({"x": bx, "y": 0.4 * bx + rng.normal(0, se), "w": 1 / se**2})
        ax = _draw(df, fit="ols", weights="w", nointercept=True)
        assert self._slope(ax) == pytest.approx(0.4, abs=0.02)
        (line,) = ax.get_lines()
        x, y = line.get_xdata(), line.get_ydata()
        assert y[0] == pytest.approx(0.4 * x[0], abs=0.02)  # through the origin

    def test_rows_without_weight_left_out(self, capsys):
        df = pd.DataFrame(
            {"x": [1.0, 2, 3, 4, 5], "y": [1.0, 2, 3, 4, 50], "w": [1, 1, 1, 1, None]}
        )
        assert self._slope(_draw(df, fit="ols", weights="w")) == pytest.approx(1.0)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(weights="w"), "add --fit"),
            (dict(weights="w", fit="lowess"), "add --fit"),
            (dict(weights="w", fit="ols", robust=True), "--robust"),
            (dict(weights="neg", fit="ols"), "negative"),
        ],
    )
    def test_errors(self, cloud_df, kwargs, match):
        fname = _tsv(cloud_df.assign(neg=-1.0))
        try:
            with pytest.raises(ValueError, match=match):
                ScatCommand().execute(_scat_args(fname, _png(), **kwargs))
        finally:
            os.unlink(fname)


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

    def test_linestyle_none_draws_points_only(self, simple_df):
        import matplotlib.pyplot as plt

        from dftk.commands.line_cmd import _line_ax

        fig, ax = plt.subplots()
        df = simple_df.assign(lo=simple_df["y"] - 1, hi=simple_df["y"] + 2)
        _line_ax(ax, df, _line_args("", "", linestyle="none", marker="o", yci="lo,hi"))
        line = ax.containers[0].lines[0]  # the data line, not the bars
        assert line.get_linestyle() == "None" and line.get_marker() == "o"
        plt.close(fig)

    @pytest.mark.parametrize("yci", ["lo", "lo,hi,x", "lo,"])
    def test_yci_needs_two_columns(self, simple_df, yci):
        fname = _tsv(simple_df)
        try:
            with pytest.raises(ValueError, match="--yci takes two column names"):
                LineCommand().execute(_line_args(fname, _png(), yci=yci))
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


# ---------------------------------------------------------------------------
# ChiCommand
# ---------------------------------------------------------------------------


class TestChiCommand:
    def test_basic_chi(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ChiCommand().execute(_chi_args(fname, out))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ChiCommand().execute(_chi_args(fname, out, groupcol=["group"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ChiCommand().execute(_chi_args(fname, out, subgraphcol=["cond"]))
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_subgraphcol_and_groupcol(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            ChiCommand().execute(
                _chi_args(fname, out, subgraphcol=["cond"], groupcol=["group"])
            )
            _assert_file_created(out)
        finally:
            os.unlink(fname)
            if os.path.exists(out):
                os.unlink(out)

    def test_ties_do_not_crash(self):
        # Heavily rounded/repeated values are the case that broke the
        # tie-unsafe leave-one-out CDF implementation.
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "x": rng.integers(0, 6, 300).astype(float),
                "y": rng.integers(0, 6, 300).astype(float),
            }
        )
        fname = _tsv(df)
        out = _png()
        try:
            ChiCommand().execute(_chi_args(fname, out))
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
                ChiCommand().execute(_chi_args(fname, out, xcol="no_such"))
        finally:
            os.unlink(fname)

    def test_missing_groupcol_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError, match="no_such"):
                ChiCommand().execute(_chi_args(fname, out, groupcol=["no_such"]))
        finally:
            os.unlink(fname)

    def test_missing_xcol_arg_raises(self, simple_df):
        fname = _tsv(simple_df)
        out = _png()
        try:
            with pytest.raises(ValueError):
                ChiCommand().execute(_chi_args(fname, out, xcol=None))
        finally:
            os.unlink(fname)
