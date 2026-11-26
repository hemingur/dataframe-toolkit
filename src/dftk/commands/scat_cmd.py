"""
dftk.commands.scat_cmd — dftk scat subcommand.

Scatter plot with optional OLS / robust regression line overlay.

Port of dfscat.py with improvements:
  - --groupcol  colour-codes points by a column
  - --subgraphcol  splits into a subplot grid
  - --size / --fontsize  publication-quality figure presets
  - --fit  overlays an OLS (or robust) regression line via fit_cmd.regress_it,
    or a LOWESS curve (--fit lowess); with --groupcol, one per group
  - --hexbin  hexagonal binning for large tables (port of dfhexb.py)
  - --sizecol  point size from a data column (was --area)
  - --colorcol  continuous colour mapping from a data column (was --color)
  - --xerr / --yerr  error bars from columns (--errscale 1.96 for 95 % CIs)
  - -w/--weights  weighted least squares for --fit ols (e.g. 1/se² for an
    inverse-variance-weighted Mendelian randomisation line)
"""

import argparse
import sys

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io, option_columns
from dftk.common.plot import (
    add_figure_arguments,
    add_group_arguments,
    add_legend_arguments,
    add_refline_arguments,
    add_xy_arguments,
    apply_ablines,
    apply_labels,
    apply_limits,
    apply_reflines,
    apply_style,
    apply_ticks,
    make_figure,
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

# ---------------------------------------------------------------------------
# Fit overlay
# ---------------------------------------------------------------------------


def _ols_curve(x, y, args, label: str, w=None):
    """Fit OLS (RLM with --robust, WLS with weights *w*) of y on x; return
    (slope, intercept, legend text)."""
    import pandas as pd

    from dftk.commands.fit_cmd import regress_it

    data = pd.DataFrame({"_x": x, "_y": y})
    if w is not None:
        data["_w"] = w
    fit_args = argparse.Namespace(
        formula="_y ~ _x" if not args.nointercept else "_y ~ _x - 1",
        weights="_w" if w is not None else None,
        robust=getattr(args, "robust", False),
    )
    res = regress_it(data, fit_args)
    if label:
        print(f"== {label}", file=sys.stderr)
    print(res.summary(), file=sys.stderr)

    slope = res.params["_x"]
    intercept = 0.0 if args.nointercept else res.params["Intercept"]
    if args.nointercept:
        text = f"{slope:+.3g} × {args.xcol}"
    else:
        text = f"{intercept:.3g}{slope:+.3g} × {args.xcol}"
    if w is not None:
        text += "  (weighted)"
    if getattr(args, "pvalue", False):
        pval = res.pvalues.get("_x", res.pvalues.iloc[-1])
        text += f"  -log10(p)={-np.log10(pval):.2g}"
    return slope, intercept, text


def _lowess_curve(x, y, args):
    """Return the LOWESS curve of y on x as (x_sorted, y_smoothed)."""
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # delta: fit only every 1% of the x range and interpolate in between.
    # Without it LOWESS is quadratic in n (10,000 points take seconds,
    # 200,000 do not finish); with it 1,000,000 points take a few seconds.
    delta = 0.01 * (x.max() - x.min())
    curve = lowess(y, x, frac=args.frac, delta=delta)
    return curve[:, 0], curve[:, 1]


def _fit_overlay(
    ax, x_series, y_series, args, color=None, label: str = "", w_series=None
):
    """Overlay the --fit line or curve on ax.

    Without *color* (no --groupcol) the fit gets its own red legend entry,
    as before.  With *color* it is drawn in the group's colour, over that
    group's x range, and the group legend already identifies it.
    *w_series*: weights for the OLS line (rows without one are left out).
    """
    import pandas as pd

    data = pd.DataFrame({"x": x_series.values, "y": y_series.values})
    if w_series is not None:
        data["w"] = pd.to_numeric(w_series, errors="coerce").values
    data = data.dropna().astype(float)
    if len(data) < 3:
        print(
            f"--fit: skipped {label or 'data'} (fewer than 3 points)", file=sys.stderr
        )
        return
    x, y = data["x"].to_numpy(), data["y"].to_numpy()

    if args.fit == "lowess":
        x_plot, y_plot = _lowess_curve(x, y, args)
        text = f"LOWESS (frac={args.frac:.2g})"
    else:
        w = data["w"].to_numpy() if w_series is not None else None
        slope, intercept, text = _ols_curve(x, y, args, label, w)
        xmin, xmax = (x.min(), x.max()) if color is not None else ax.get_xlim()
        x_plot = np.linspace(xmin, xmax, 200)
        y_plot = intercept + slope * x_plot

    if color is None:
        ax.plot(x_plot, y_plot, "r--", linewidth=1)
        ax.legend(
            handles=[mlines.Line2D([], [], color="red", linestyle="--", label=text)],
            loc=0,
        )
    else:
        ax.plot(x_plot, y_plot, "--", color=color, linewidth=1.2)


def _hexbin_ax(ax, df, args):
    """Draw a hexagonal-bin density plot (or mean of --colorcol per hexagon)."""
    import matplotlib.pyplot as plt

    cols = [args.xcol, args.ycol] + ([args.colorcol] if args.colorcol else [])
    data = df[cols].dropna()
    for flag, col in (("logx", args.xcol), ("logy", args.ycol)):
        if getattr(args, flag, False):
            positive = data[col] > 0
            if not positive.all():
                print(
                    f"--hexbin --{flag}: dropped {(~positive).sum()} rows with "
                    f"{col} <= 0",
                    file=sys.stderr,
                )
                data = data[positive]

    kw = dict(
        gridsize=args.gridsize,
        mincnt=1,
        cmap="viridis",
        # bin on the log scale, so hexagons are regular on log axes
        xscale="log" if getattr(args, "logx", False) else "linear",
        yscale="log" if getattr(args, "logy", False) else "linear",
    )
    if args.hexlog:
        kw["bins"] = "log"
    if args.colorcol:
        kw["C"] = data[args.colorcol]
        kw["reduce_C_function"] = np.mean
        label = f"mean {args.colorcol}"
    else:
        label = "count"
    hb = ax.hexbin(data[args.xcol], data[args.ycol], **kw)
    plt.colorbar(hb, ax=ax, label=label)


def _share_hexbin_colours(axes) -> None:
    """Give every --subgraphcol panel the same colour scale, so they compare."""
    from matplotlib.collections import PolyCollection

    hexbins = [
        c
        for row in axes
        for ax in row
        for c in ax.collections
        if isinstance(c, PolyCollection) and c.get_array() is not None
    ]
    values = [hb.get_array() for hb in hexbins if hb.get_array().size]
    if not values:
        return
    vmin = min(float(v.min()) for v in values)
    vmax = max(float(v.max()) for v in values)
    for hb in hexbins:
        hb.set_clim(vmin, vmax)


# ---------------------------------------------------------------------------
# Single-axes scatter (reused per subplot)
# ---------------------------------------------------------------------------


def _errorbars(ax, df, args, color) -> None:
    """Draw --xerr / --yerr bars (times --errscale) under the points."""
    xerr = getattr(args, "xerr", None)
    yerr = getattr(args, "yerr", None)
    if not (xerr or yerr):
        return
    scale = getattr(args, "errscale", 1.0)
    ax.errorbar(
        df[args.xcol],
        df[args.ycol],
        xerr=df[xerr].to_numpy(dtype=float) * scale if xerr else None,
        yerr=df[yerr].to_numpy(dtype=float) * scale if yerr else None,
        fmt="none",
        ecolor=color,
        elinewidth=0.8,
        alpha=0.7,
        zorder=1,
    )


def _weights(df, args):
    w = getattr(args, "weights", None)
    return df[w] if w else None


def _scatter_ax(ax, df, args, title_suffix: str = ""):
    """Draw scatter on a single Axes, with optional --groupcol colouring."""
    import matplotlib.pyplot as plt

    xcol = args.xcol
    ycol = args.ycol
    legendloc = tuple(args.legendloc) if args.legendloc else "best"

    scatter_kw = dict(marker=getattr(args, "marker", "o"), alpha=0.7)

    if getattr(args, "hexbin", False):
        _hexbin_ax(ax, df, args)
        if getattr(args, "fit", None):
            _fit_overlay(ax, df[xcol], df[ycol], args)
    elif args.groupcol is not None:
        for groupname, gdf in subgraph_groups(
            df, args.groupcol, getattr(args, "groupcolorder", None)
        ):
            label = make_grouplabel(
                groupname, args.groupcol, getattr(args, "groupcolformat", None)
            )
            kw = dict(scatter_kw)
            if args.sizecol:
                kw["s"] = gdf[args.sizecol]
            points = ax.scatter(gdf[xcol], gdf[ycol], label=label, zorder=2, **kw)
            color = points.get_facecolor()[0]
            _errorbars(ax, gdf, args, color)
            if getattr(args, "fit", None):
                _fit_overlay(
                    ax, gdf[xcol], gdf[ycol], args, color=color, label=label,
                    w_series=_weights(gdf, args),
                )  # fmt: skip
        ax.legend(loc=legendloc, title=args.legendtitle)
    else:
        kw = dict(scatter_kw)
        if args.sizecol:
            kw["s"] = df[args.sizecol]
        if args.colorcol:
            sc = ax.scatter(
                df[xcol], df[ycol], c=df[args.colorcol], cmap="viridis", zorder=2,
                **kw,
            )  # fmt: skip
            plt.colorbar(sc, ax=ax, label=args.colorcol)
            _errorbars(ax, df, args, "grey")
        else:
            label = args.legend or ""
            points = ax.scatter(
                df[xcol], df[ycol], label=label if label else None, zorder=2, **kw
            )
            _errorbars(ax, df, args, points.get_facecolor()[0])
            if label:
                ax.legend(loc=legendloc, title=args.legendtitle)

        if getattr(args, "fit", None):
            _fit_overlay(ax, df[xcol], df[ycol], args, w_series=_weights(df, args))

    apply_reflines(ax, args)
    apply_limits(ax, args)
    apply_ablines(ax, args)
    apply_ticks(ax, args)
    apply_labels(ax, args, default_xlabel=xcol or "", default_ylabel=ycol or "")
    if title_suffix:
        existing = ax.get_title()
        ax.set_title(f"{existing}  [{title_suffix}]" if existing else title_suffix)

    ax.grid(True, linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
EXAMPLES
--------
Basic scatter:

  dftk scat data.tsv -x weight -y height
  dftk dataset tips -o | dftk scat -x total_bill -y tip

Colour by group:

  dftk dataset tips -o | dftk scat -x total_bill -y tip -g day

Subplot grid by time-of-day, colour by smoker:

  dftk dataset tips -o | dftk scat -x total_bill -y tip \\
      --subgraphcol time -g smoker

Fit OLS line, or one per group in the group's colour:

  dftk dataset tips -o | dftk scat -x total_bill -y tip --fit
  dftk dataset tips -o | dftk scat -x total_bill -y tip -g smoker --fit

LOWESS curve (local regression; smaller --frac follows the data more
closely). At the ends of the x range, where few points are left, the curve
can overshoot the data; a larger --frac or --xlim helps:

  dftk dataset tips -o | dftk scat -x total_bill -y tip --fit lowess

Lines y = A + B*x, here 15 % and 20 % tips, on every panel (bare --abline
is y = x; a negative intercept is written --abline=-1:2):

  dftk dataset tips -o | dftk scat -x total_bill -y tip --subgraphcol time \\
      --abline 0:0.15:15% 0:0.2:20%

Hexagonal binning for large tables (54,000 diamonds), log colour scale and
log axes; hexagons are binned on the log scale:

  dftk dataset diamonds -o | dftk scat -x carat -y price \\
      --hexbin --hexlog --logx --logy

Colour each hexagon by the mean of another column instead of the count:

  dftk dataset diamonds -o | dftk scat -x carat -y price \\
      --hexbin --colorcol depth --gridsize 50

One hexbin panel per group, all on the same colour scale:

  dftk dataset diamonds -o | dftk scat -x carat -y price \\
      --hexbin --hexlog --subgraphcol cut --gridsize 40

Publication figure (Nature single column, PDF):

  dftk dataset tips -o | dftk scat -x total_bill -y tip \\
      --size single --fontsize publication -f fig.pdf

Bubble chart (point size from column):

  dftk dataset tips -o | dftk scat -x total_bill -y tip --sizecol size

Continuous colour mapping:

  dftk dataset tips -o | dftk scat -x total_bill -y tip --colorcol tip

Error bars from standard-error columns, as 95 % intervals (mean tip
against mean bill per party size):

  dftk dataset tips -o | dftk stat -c total_bill tip -g size -o \\
      | dftk pivot -i size -g name -v mean sem -o \\
      | dftk scat -x mean_total_bill -y mean_tip --xerr sem_total_bill \\
          --yerr sem_tip --errscale 1.96

Mendelian-randomisation style: effect on the outcome (by) against effect
on the exposure (bx), each ± its standard error, and the inverse-variance-
weighted line through the origin (weights 1/se_y²; 30 made-up instruments
with a true slope of 0.4):

  dftk randvar -n 30 -d bx --dist uniform --parameters loc:0.05,scale:0.45 \\
          --randomseed 1 -o \\
      | dftk randvar -d noise --dist norm --parameters scale:0.02 --randomseed 2 -o \\
      | dftk eval -f "se_x = 0.01 + 0.04 * bx" -f "se_y = 0.01 + 0.06 * bx" \\
          -f "by = 0.4 * bx + noise" -f "w = 1 / se_y**2" -o \\
      | dftk scat -x bx -y by --xerr se_x --yerr se_y --fit -w w --nointercept
"""


class ScatCommand(BaseCommand):
    name = "scat"
    help = "Scatter plot with optional regression line overlay."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_xy_arguments(parser)
        add_figure_arguments(parser)
        add_refline_arguments(parser, sloped=True)
        add_group_arguments(parser)
        add_legend_arguments(parser)

        s = parser.add_argument_group("scatter options")
        s.add_argument(
            "-m",
            "--marker",
            default="o",
            metavar="MARKER",
            help="Matplotlib marker style (default: o).",
        )
        s.add_argument(
            "--sizecol",
            default=None,
            metavar="COL",
            help="Column to use for point sizes (bubble chart).",
        )
        s.add_argument(
            "--colorcol",
            default=None,
            metavar="COL",
            help="Column for continuous colour mapping.",
        )

        e = parser.add_argument_group("error bars")
        e.add_argument(
            "--xerr",
            default=None,
            metavar="COL",
            help="Column of symmetric ± errors on x (e.g. a standard error).",
        )
        e.add_argument(
            "--yerr",
            default=None,
            metavar="COL",
            help="Column of symmetric ± errors on y.",
        )
        e.add_argument(
            "--errscale",
            type=float,
            default=1.0,
            metavar="Z",
            help="Multiply --xerr/--yerr by Z (default: 1; 1.96 turns standard "
            "errors into 95%% confidence intervals).",
        )

        h = parser.add_argument_group("hexbin options")
        h.add_argument(
            "--hexbin",
            action="store_true",
            help="Hexagonal binning instead of points, for large tables: "
            "colour shows the number of points per hexagon, or with "
            "--colorcol the mean of that column. Not with -g or --sizecol.",
        )
        h.add_argument(
            "--gridsize",
            type=int,
            default=100,
            metavar="N",
            help="Number of hexagons across the x axis (default: 100).",
        )
        h.add_argument(
            "--hexlog",
            action="store_true",
            help="Log colour scale, so sparse areas stay visible.",
        )

        f = parser.add_argument_group("fit options")
        f.add_argument(
            "--fit",
            nargs="?",
            const="ols",
            default=None,
            choices=["ols", "lowess"],
            help="Overlay a fit: ols (the default for a bare --fit; a "
            "regression line, summary to stderr) or lowess (a smooth curve). "
            "With -g, one per group in the group's colour.",
        )
        f.add_argument(
            "--frac",
            type=float,
            default=2 / 3,
            metavar="F",
            help="Share of the data used for each local fit in --fit lowess "
            "(default: 2/3). Smaller values follow the data more closely.",
        )
        f.add_argument(
            "-r",
            "--robust",
            action="store_true",
            help="Use robust regression (RLM) for --fit ols.",
        )
        f.add_argument(
            "--nointercept", action="store_true", help="Suppress intercept in --fit."
        )
        f.add_argument(
            "-w",
            "--weights",
            default=None,
            metavar="COL",
            help="Weights for --fit ols (weighted least squares), e.g. 1/se² "
            "made with eval; rows without a weight are left out of the fit.",
        )
        f.add_argument(
            "--pvalue",
            action="store_true",
            help="Add −log₁₀(p) to the fit legend label.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if not args.xcol or not args.ycol:
            raise ValueError("--xcol (-x) and --ycol (-y) are required.")

        df = io.read(args, columns=option_columns(args))
        check_cols(df, [args.xcol, args.ycol], "-x/-y")
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        if args.sizecol:
            check_cols(df, [args.sizecol], "--sizecol")
        if args.colorcol:
            check_cols(df, [args.colorcol], "--colorcol")
        if args.hexbin:
            if args.groupcol:
                raise ValueError(
                    "--hexbin cannot be combined with -g/--groupcol: overlaid "
                    "densities cannot be told apart. Use --subgraphcol instead."
                )
            if args.sizecol:
                raise ValueError("--hexbin cannot be combined with --sizecol.")
        for opt, col in (("--xerr", args.xerr), ("--yerr", args.yerr)):
            if col:
                check_cols(df, [col], opt)
                values = pd.to_numeric(df[col], errors="coerce")
                if (values < 0).any():
                    raise ValueError(f"{opt} column {col!r} has negative values.")
        if args.errscale <= 0:
            raise ValueError(f"--errscale must be above 0, not {args.errscale:g}.")
        if args.weights:
            check_cols(df, [args.weights], "-w/--weights")
            if args.fit != "ols":
                raise ValueError("-w/--weights weights the --fit ols line; add --fit.")
            if args.robust:
                raise ValueError("-w/--weights can't be combined with --robust.")
            if (pd.to_numeric(df[args.weights], errors="coerce") < 0).any():
                raise ValueError(
                    f"-w/--weights column {args.weights!r} has negative values."
                )
        if args.hexbin and (args.xerr or args.yerr):
            raise ValueError("--hexbin draws no points to put error bars on.")
        if args.fit == "lowess" and not 0 < args.frac <= 1:
            raise ValueError(f"--frac must be in (0, 1], got {args.frac}")

        # Use non-interactive backend when saving to file
        if args.file:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)

        apply_style(args)

        if args.subgraphcol is not None:
            groups = subgraph_groups(
                df, args.subgraphcol, getattr(args, "subgraphorder", None)
            )
            nrows, ncols = subgraph_layout(len(groups), args.ncols)
            fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
            for idx, (groupname, gdf) in enumerate(groups):
                ax = axes[idx // ncols][idx % ncols]
                suffix = make_grouplabel(
                    groupname, args.subgraphcol, getattr(args, "subgraphformat", None)
                )
                _scatter_ax(ax, gdf, args, title_suffix=suffix)
            # Hide unused axes
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            if args.hexbin:
                _share_hexbin_colours(axes)
            fig.tight_layout()
        else:
            fig, axes = make_figure(args)
            _scatter_ax(axes[0][0], df, args)
            fig.tight_layout()

        save_or_show(fig, args)
