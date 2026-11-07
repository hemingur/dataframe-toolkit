"""
dftk.commands.pairs_cmd — dftk pairs subcommand.

Scatter-plot matrix: every pair of columns as a small scatter plot, with each
column's distribution (KDE or histogram) on the diagonal.  The upper triangle
repeats the scatter plots mirrored, or shows the correlation (--upper corr).
"""

import argparse
import fnmatch

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io
from dftk.common.plot import (
    _DEFAULT_SIZE,
    WONG_PALETTE,
    add_figure_arguments,
    apply_style,
    parse_figsize,
    save_or_show,
)

_PANEL_INCHES = 2.2  # default size of one panel
_MAX_INCHES = 12.0  # default figure size cap


def _columns(df: pd.DataFrame, patterns: list[str] | None, exclude) -> list[str]:
    """-c names and globs (default: every numeric column), in that order."""
    numeric = [
        c
        for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]
    if not patterns:
        cols = numeric
    else:
        cols = []
        for pattern in patterns:
            if set("*?[") & set(pattern):
                matched = fnmatch.filter(numeric, pattern)
                if not matched:
                    raise ValueError(f"-c: {pattern!r} matches no numeric column")
            else:
                check_cols(df, [pattern], "-c/--cols")
                matched = [pattern]
            cols += [c for c in matched if c not in cols]
    if len(cols) < 2:
        raise ValueError(f"pairs needs at least two numeric columns, got {cols}")
    return cols


def _groups(df: pd.DataFrame, groupcol: str | None, order: list[str] | None):
    """[(label, sub-DataFrame)], in --groupcolorder, category or first-seen order."""
    if groupcol is None:
        return [(None, df)]
    col = df[groupcol]
    if order:
        values = [v for v in col.unique() if str(v) in order]
        values.sort(key=lambda v: order.index(str(v)))
        values += [v for v in col.unique() if str(v) not in order]
    elif isinstance(col.dtype, pd.CategoricalDtype):
        values = [v for v in col.cat.categories if (col == v).any()]
    else:
        values = list(col.dropna().unique())
    return [(str(v), df[col == v]) for v in values]


def _diag(ax, groups, col: str, kind: str, colors) -> None:
    """One column's distribution: a KDE (or histogram) per group."""
    from scipy.stats import gaussian_kde

    values = np.concatenate([g[col].dropna().to_numpy(float) for _, g in groups])
    if values.size == 0:
        return
    lo, hi = values.min(), values.max()
    pad = 0.05 * (hi - lo or 1.0)
    grid = np.linspace(lo - pad, hi + pad, 200)
    bins = np.histogram_bin_edges(values, bins="auto")
    for (label, g), color in zip(groups, colors, strict=False):
        v = g[col].dropna().to_numpy(float)
        if kind == "hist":
            ax.hist(v, bins=bins, color=color, alpha=0.5, label=label)
        elif v.size > 1 and np.ptp(v) > 0:
            density = gaussian_kde(v)(grid)
            ax.fill_between(grid, density, color=color, alpha=0.25)
            ax.plot(grid, density, color=color, label=label)
    ax.set_yticks([])


def _scatter(ax, groups, xcol: str, ycol: str, args, colors) -> None:
    for (label, g), color in zip(groups, colors, strict=False):
        ax.scatter(
            g[xcol],
            g[ycol],
            s=args.markersize,
            alpha=args.alpha,
            color=color,
            label=label,
            linewidths=0,
            rasterized=True,  # keeps a PDF small with many points
        )


def _corr_text(ax, df: pd.DataFrame, xcol: str, ycol: str, method: str, cmap):
    """The correlation of the pair, on a cell coloured by its value."""
    pair = df[[xcol, ycol]].dropna()
    r = pair[xcol].corr(pair[ycol], method=method) if len(pair) > 1 else np.nan
    if np.isfinite(r):
        ax.set_facecolor(cmap((r + 1) / 2))
        ax.text(
            0.5,
            0.5,
            f"{r:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize="large",
            color="white" if abs(r) > 0.6 else "black",
        )
    ax.set_xticks([])
    ax.set_yticks([])


def _figsize(args: argparse.Namespace, k: int) -> tuple[float, float]:
    if args.size != _DEFAULT_SIZE:
        return parse_figsize(args.size)
    side = min(_PANEL_INCHES * k, _MAX_INCHES)
    return side, side


_EPILOG = """\
LAYOUT
------
Row i, column j shows column j (x) against column i (y); the diagonal shows
each column's distribution, a KDE by default.  All numeric columns are used
unless -c names some (globs allowed, in the order given):

  dftk pairs data.tsv -c x y z
  dftk dataset iris -o | dftk pairs -f iris_pairs.png
  dftk dataset penguins -o | dftk pairs -c 'bill*' 'flipper*' body_mass_g \\
      -f penguin_pairs.png

GROUPS
------
-g colours points by group, with one KDE per group on the diagonal:

  dftk dataset iris -o | dftk pairs -g species -f iris_species.png

UPPER TRIANGLE AND DIAGONAL
---------------------------
--upper corr writes the correlation (--method pearson, spearman or
kendall) in the upper cells, coloured blue (negative) to red (positive),
instead of repeating the scatter plots.  --diag hist draws histograms:

  dftk dataset iris -o | dftk pairs -g species --upper corr --diag hist \\
      -f iris_corr.png

LARGE TABLES
------------
Points are rasterized, so a PDF stays small.  With many points, lower
--alpha and --markersize, or plot a sample:

  dftk dataset diamonds -o | dftk sample -n 5000 --randomseed 1 -o \\
    | dftk pairs -c carat depth price --alpha 0.2 --upper corr \\
        --method spearman -f diamonds_pairs.png
"""


class PairsCommand(BaseCommand):
    name = "pairs"
    help = "Scatter-plot matrix of column pairs, with KDEs on the diagonal."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)

        g = parser.add_argument_group("pairs")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            default=None,
            metavar="COL",
            help="Columns to plot, globs allowed (default: all numeric columns).",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            default=None,
            metavar="COL",
            help="Colour by this column; one KDE per group on the diagonal.",
        )
        g.add_argument(
            "--groupcolorder",
            nargs="+",
            default=None,
            metavar="VAL",
            help="Group order for -g; values not listed come last.",
        )
        g.add_argument(
            "--diag",
            choices=["kde", "hist"],
            default="kde",
            help="Distribution on the diagonal (default: kde).",
        )
        g.add_argument(
            "--upper",
            choices=["scatter", "corr"],
            default="scatter",
            help="Upper triangle: mirrored scatter plots, or the correlation "
            "(default: scatter).",
        )
        g.add_argument(
            "--method",
            choices=["pearson", "spearman", "kendall"],
            default="pearson",
            help="Correlation for --upper corr (default: pearson).",
        )
        g.add_argument(
            "--alpha",
            type=float,
            default=0.6,
            help="Point opacity, 0-1 (default: 0.6).",
        )
        g.add_argument(
            "--markersize",
            type=float,
            default=8.0,
            help="Point size in points^2 (default: 8).",
        )

        add_figure_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        if args.groupcol is not None:
            check_cols(df, [args.groupcol], "-g/--groupcol")
        cols = _columns(df, args.cols, exclude={args.groupcol})
        groups = _groups(df, args.groupcol, args.groupcolorder)
        colors = [WONG_PALETTE[i % len(WONG_PALETTE)] for i in range(len(groups))]
        if args.groupcol is None:
            colors = [WONG_PALETTE[4]]  # blue

        if args.file:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if getattr(args, "usetex", False):
            plt.rc("text", usetex=True)

        apply_style(args)
        k = len(cols)
        fig, axes = plt.subplots(
            k, k, figsize=_figsize(args, k), squeeze=False, layout="constrained"
        )
        cmap = plt.get_cmap("RdBu_r")

        for i, ycol in enumerate(cols):
            for j, xcol in enumerate(cols):
                ax = axes[i][j]
                if i == j:
                    _diag(ax, groups, xcol, args.diag, colors)
                elif i < j and args.upper == "corr":
                    _corr_text(ax, df, xcol, ycol, args.method, cmap)
                else:
                    _scatter(ax, groups, xcol, ycol, args, colors)
                # Labels on the outer edge only; columns share x, rows share y
                if i == k - 1:
                    ax.set_xlabel(xcol)
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel(ycol)
                if j != 0 and i != j:
                    ax.tick_params(labelleft=False)

        # Same x range down each column, same y range along each row
        for j in range(k):
            lim = axes[j][j].get_xlim()
            for i in range(k):
                if not (i < j and args.upper == "corr"):
                    axes[i][j].set_xlim(lim)
        for i in range(k):
            others = [
                axes[i][j]
                for j in range(k)
                if j != i and not (i < j and args.upper == "corr")
            ]
            if others:
                lims = [ax.get_ylim() for ax in others]
                for ax in others:
                    ax.set_ylim(min(lo for lo, _ in lims), max(hi for _, hi in lims))

        if args.groupcol is not None:
            handles, labels = axes[0][0].get_legend_handles_labels()
            if not handles:
                handles, labels = axes[-1][0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                title=args.groupcol,
                loc="outside right center",
            )
        if args.title:
            fig.suptitle(args.title)
        fig.align_ylabels(axes[:, 0])
        save_or_show(fig, args)
