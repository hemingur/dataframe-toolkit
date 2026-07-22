"""
dftk.commands.line_cmd — dftk line subcommand.

Line plot with optional error bars, confidence intervals, and regression
line overlay.

Port of dfline.py with improvements:
  - --groupcol  draws one line per group value (colours from Wong palette)
  - --subgraphcol  splits into a subplot grid
  - --size / --fontsize  publication-quality figure presets
  - --yerr / --yci  symmetric or asymmetric error bars
  - --fit  overlays an OLS regression line (uses fit_cmd.regress_it)
  - --drawstyle  step / default for staircase plots
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
# Error bar helper
# ---------------------------------------------------------------------------


def _resolve_yerr(df, args):
    """Return yerr array for df, or None.  Handles --yerr and --yci."""
    if args.yerr is not None:
        err = df[args.yerr].values
        return [err, err]
    if args.yci is not None:
        lo_col, hi_col = args.yci.split(",")
        lo = (df[args.ycol] - df[lo_col]).values
        hi = (df[hi_col] - df[args.ycol]).values
        return [lo, hi]
    return None


# ---------------------------------------------------------------------------
# Fit overlay (shared logic with scat)
# ---------------------------------------------------------------------------


def _fit_overlay(ax, x_series, y_series, args):
    import pandas as pd

    from dftk.commands.fit_cmd import regress_it

    df_fit = pd.DataFrame({"_x": x_series.values, "_y": y_series.values}).dropna()
    fit_args = argparse.Namespace(
        formula="_y ~ _x" if not args.nointercept else "_y ~ _x - 1",
        weights=getattr(args, "weights", None),
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
# Single-axes line plot
# ---------------------------------------------------------------------------


def _line_ax(ax, df, args, title_suffix: str = ""):
    xcol = args.xcol
    ycol = args.ycol
    legendloc = tuple(args.legendloc) if args.legendloc else "best"
    marker = getattr(args, "marker", "")
    drawstyle = getattr(args, "drawstyle", "default")

    if args.groupcol is not None:
        for groupname, gdf in subgraph_groups(
            df, args.groupcol, getattr(args, "groupcolorder", None)
        ):
            label = make_grouplabel(
                groupname, args.groupcol, getattr(args, "groupcolformat", None)
            )
            yerr = _resolve_yerr(gdf, args)
            ax.errorbar(
                gdf[xcol],
                gdf[ycol],
                yerr=yerr,
                marker=marker,
                drawstyle=drawstyle,
                label=label,
                capsize=3,
            )
        ax.legend(loc=legendloc, title=args.legendtitle)
    else:
        yerr = _resolve_yerr(df, args)
        label = args.legend or ""
        ax.errorbar(
            df[xcol],
            df[ycol],
            yerr=yerr,
            marker=marker,
            drawstyle=drawstyle,
            label=label if label else None,
            capsize=3,
        )
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
Basic line plot:

  dftk line data.tsv -x time -y value

Multiple lines by group:

  dftk line data.tsv -x time -y value -g group

Symmetric error bars:

  dftk line data.tsv -x time -y mean --yerr stderr

Asymmetric confidence intervals:

  dftk line data.tsv -x time -y mean --yci cilo,cihi

Subplot grid by condition:

  dftk line data.tsv -x time -y value --subgraphcol condition -g group

Staircase / step plot:

  dftk line data.tsv -x time -y value --drawstyle steps-mid

Fit and overlay regression line:

  dftk line data.tsv -x time -y value --fit

Publication figure (Nature double column, PDF):

  dftk line data.tsv -x time -y value --size double --fontsize publication -f fig.pdf
"""


class LineCommand(BaseCommand):
    name = "line"
    help = "Line plot with optional error bars and regression overlay."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_xy_arguments(parser)
        add_figure_arguments(parser)
        add_group_arguments(parser)
        add_legend_arguments(parser)

        s = parser.add_argument_group("line options")
        s.add_argument(
            "-m",
            "--marker",
            default="",
            metavar="MARKER",
            help="Matplotlib marker style (default: none).",
        )
        s.add_argument(
            "--drawstyle",
            default="default",
            metavar="STYLE",
            help="Matplotlib drawstyle, e.g. steps-mid (default: default).",
        )

        e = parser.add_argument_group("error bars (mutually exclusive)")
        mx = e.add_mutually_exclusive_group()
        mx.add_argument(
            "--yerr",
            default=None,
            metavar="COL",
            help="Column of symmetric ±error values.",
        )
        mx.add_argument(
            "--yci",
            default=None,
            metavar="CILO,CIHI",
            help="Two column names for asymmetric confidence interval "
            "(comma-separated, e.g. cilo,cihi).",
        )

        f = parser.add_argument_group("fit options")
        f.add_argument(
            "--fit",
            action="store_true",
            help="Overlay an OLS regression line (no grouping). "
            "Prints summary to stderr.",
        )
        f.add_argument(
            "-r",
            "--robust",
            action="store_true",
            help="Use robust regression (RLM) for --fit.",
        )
        f.add_argument(
            "-w",
            "--weights",
            default=None,
            metavar="COL",
            help="Weight column for WLS fit.",
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
        if args.yerr:
            check_cols(df, [args.yerr], "--yerr")
        if args.yci:
            lo, hi = args.yci.split(",")
            check_cols(df, [lo.strip(), hi.strip()], "--yci")

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
                _line_ax(ax, gdf, args, title_suffix=suffix)
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
        else:
            fig, axes = make_figure(args)
            _line_ax(axes[0][0], df, args)
            fig.tight_layout()

        save_or_show(fig, args)
