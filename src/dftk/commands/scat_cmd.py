"""
dftk.commands.scat_cmd — dftk scat subcommand.

Scatter plot with optional OLS / robust regression line overlay.

Port of dfscat.py with improvements:
  - --groupcol  colour-codes points by a column
  - --subgraphcol  splits into a subplot grid
  - --size / --fontsize  publication-quality figure presets
  - --fit  overlays an OLS (or robust) regression line via fit_cmd.regress_it
  - --sizecol  point size from a data column (was --area)
  - --colorcol  continuous colour mapping from a data column (was --color)
"""

import argparse
import sys

import matplotlib.lines as mlines
import numpy as np

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io
from dftk.common.plot import (
    add_figure_arguments,
    add_group_arguments,
    add_legend_arguments,
    add_xy_arguments,
    apply_labels,
    apply_limits,
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


def _fit_overlay(ax, x_series, y_series, args):
    """Fit a regression line and overlay it on ax. Prints summary to stdout."""
    import pandas as pd

    from dftk.commands.fit_cmd import regress_it

    df_fit = pd.DataFrame({"_x": x_series.values, "_y": y_series.values}).dropna()
    fit_args = argparse.Namespace(
        formula="_y ~ _x" if not args.nointercept else "_y ~ _x - 1",
        weights=None,
        robust=getattr(args, "robust", False),
    )
    res = regress_it(df_fit, fit_args)
    print(res.summary(), file=sys.stderr)

    xmin, xmax = ax.get_xlim()
    X_plot = np.linspace(xmin, xmax, 200)

    if args.nointercept:
        slope = res.params["_x"]
        Y_plot = slope * X_plot
        label_text = f"{slope:+.3g} × {args.xcol}"
    else:
        intercept = res.params["Intercept"]
        slope = res.params["_x"]
        Y_plot = intercept + slope * X_plot
        label_text = f"{intercept:.3g}{slope:+.3g} × {args.xcol}"

    if getattr(args, "pvalue", False):
        pval = res.pvalues.get("_x", res.pvalues.iloc[-1])
        label_text += f"  -log10(p)={-np.log10(pval):.2g}"

    ax.plot(X_plot, Y_plot, "r--", linewidth=1)
    ax.legend(
        handles=[mlines.Line2D([], [], color="red", linestyle="--", label=label_text)],
        loc=0,
    )


# ---------------------------------------------------------------------------
# Single-axes scatter (reused per subplot)
# ---------------------------------------------------------------------------


def _scatter_ax(ax, df, args, title_suffix: str = ""):
    """Draw scatter on a single Axes, with optional --groupcol colouring."""
    import matplotlib.pyplot as plt

    xcol = args.xcol
    ycol = args.ycol
    legendloc = tuple(args.legendloc) if args.legendloc else "best"

    scatter_kw = dict(marker=getattr(args, "marker", "o"), alpha=0.7)

    if args.groupcol is not None:
        for groupname, gdf in subgraph_groups(
            df, args.groupcol, getattr(args, "groupcolorder", None)
        ):
            label = make_grouplabel(
                groupname, args.groupcol, getattr(args, "groupcolformat", None)
            )
            kw = dict(scatter_kw)
            if args.sizecol:
                kw["s"] = gdf[args.sizecol]
            ax.scatter(gdf[xcol], gdf[ycol], label=label, **kw)
        ax.legend(loc=legendloc, title=args.legendtitle)
    else:
        kw = dict(scatter_kw)
        if args.sizecol:
            kw["s"] = df[args.sizecol]
        if args.colorcol:
            sc = ax.scatter(
                df[xcol], df[ycol], c=df[args.colorcol], cmap="viridis", **kw
            )
            plt.colorbar(sc, ax=ax, label=args.colorcol)
        else:
            label = args.legend or ""
            ax.scatter(df[xcol], df[ycol], label=label if label else None, **kw)
            if label:
                ax.legend(loc=legendloc, title=args.legendtitle)

    if getattr(args, "fit", False) and args.groupcol is None:
        _fit_overlay(ax, df[xcol], df[ycol], args)

    apply_limits(ax, args)
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

Colour by group:

  dftk scat data.tsv -x x -y y -g group

Subplot grid by condition, colour by treatment:

  dftk scat data.tsv -x x -y y --subgraphcol condition -g treatment

Fit OLS line:

  dftk scat data.tsv -x x -y y --fit

Publication figure (Nature single column, PDF):

  dftk scat data.tsv -x x -y y --size single --fontsize publication -f fig.pdf

Bubble chart (point size from column):

  dftk scat data.tsv -x x -y y --sizecol area_col

Continuous colour mapping:

  dftk scat data.tsv -x x -y y --colorcol score
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

        f = parser.add_argument_group("fit options")
        f.add_argument(
            "--fit",
            action="store_true",
            help="Overlay an OLS regression line. Prints summary to stderr.",
        )
        f.add_argument(
            "-r",
            "--robust",
            action="store_true",
            help="Use robust regression (RLM) for --fit.",
        )
        f.add_argument(
            "--nointercept", action="store_true", help="Suppress intercept in --fit."
        )
        f.add_argument(
            "--pvalue",
            action="store_true",
            help="Add −log₁₀(p) to the fit legend label.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if not args.xcol or not args.ycol:
            raise ValueError("--xcol (-x) and --ycol (-y) are required.")

        df = io.read(args)
        check_cols(df, [args.xcol, args.ycol], "-x/-y")
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        if args.sizecol:
            check_cols(df, [args.sizecol], "--sizecol")
        if args.colorcol:
            check_cols(df, [args.colorcol], "--colorcol")

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
            fig.tight_layout()
        else:
            fig, axes = make_figure(args)
            _scatter_ax(axes[0][0], df, args)
            fig.tight_layout()

        save_or_show(fig, args)
