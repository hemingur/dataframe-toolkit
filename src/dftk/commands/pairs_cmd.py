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
            if set("*?[") & set(pattern) and pattern not in numeric:
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
        present = col.dropna().unique()
        values = [v for v in present if str(v) in order]
        values.sort(key=lambda v: order.index(str(v)))
        values += [v for v in present if str(v) not in order]
    elif isinstance(col.dtype, pd.CategoricalDtype):
        values = [v for v in col.cat.categories if (col == v).any()]
    else:
        values = list(col.dropna().unique())
    return [(str(v), df[col == v]) for v in values]


def _limits(df: pd.DataFrame, col: str) -> tuple[float, float] | None:
    """Axis range of *col*: its data range plus 5% (None if all missing)."""
    values = df[col].dropna().to_numpy(float)
    if values.size == 0:
        return None
    lo, hi = values.min(), values.max()
    pad = 0.05 * (hi - lo or abs(lo) or 1.0)  # a constant column too
    return lo - pad, hi + pad


def _diag(ax, groups, col: str, kind: str, colors) -> None:
    """One column's distribution: a KDE (or histogram) per group.

    With groups, each KDE is scaled by the group's share of the rows, so the
    areas compare as the histogram's counts do.
    """
    from scipy.stats import gaussian_kde

    ax.set_yticks([])
    data = [g[col].dropna().to_numpy(float) for _, g in groups]
    values = np.concatenate(data)
    if values.size == 0:
        return
    if kind == "hist":
        bins = np.histogram_bin_edges(values, bins="auto")
        for (label, _), v, color in zip(groups, data, colors, strict=False):
            ax.hist(v, bins=bins, color=color, alpha=0.5, label=label)
    else:
        kdes = [
            (label, v, gaussian_kde(v), color)
            for (label, _), v, color in zip(groups, data, colors, strict=False)
            if v.size > 1 and np.ptp(v) > 0
        ]
        if kdes:
            # Run the curves 3 bandwidths past the data, where they reach ~0
            bw = max(float(np.sqrt(kde.covariance[0, 0])) for _, _, kde, _ in kdes)
            grid = np.linspace(values.min() - 3 * bw, values.max() + 3 * bw, 200)
            for label, v, kde, color in kdes:
                density = kde(grid) * v.size / values.size
                ax.fill_between(grid, density, color=color, alpha=0.25)
                ax.plot(grid, density, color=color, label=label)


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


def _r(df: pd.DataFrame, xcol: str, ycol: str, method: str) -> float:
    pair = df[[xcol, ycol]].dropna()
    return pair[xcol].corr(pair[ycol], method=method) if len(pair) > 1 else np.nan


def _corr_text(ax, df, groups, xcol: str, ycol: str, method: str, cmap, colors):
    """The correlation of the pair; with groups, also r within each group.

    Without groups the cell is coloured by r.  With groups, r over all rows
    can differ even in sign from r within every group (Simpson's paradox),
    so each group's r is written below it, in the group's colour.
    """
    r = _r(df, xcol, ycol, method)
    if len(groups) == 1:
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
    else:
        lines = [("all", r, "black")] + [
            (label, _r(g, xcol, ycol, method), color)
            for (label, g), color in zip(groups, colors, strict=False)
        ]
        step = 1 / (len(lines) + 1)
        for n, (label, value, color) in enumerate(lines, start=1):
            text = f"{value:.2f}" if np.isfinite(value) else "-"
            ax.text(
                0.5,
                1 - n * step,
                f"{label}: {text}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize="small" if len(lines) <= 4 else "x-small",
                color=color,
                fontweight="bold" if n == 1 else "normal",
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
-g colours points by group, with one KDE per group on the diagonal.  Each
KDE is scaled by its group's share of the rows, so a small group has a
small curve, as its histogram (--diag hist) would:

  dftk dataset iris -o | dftk pairs -g species -f iris_species.png

UPPER TRIANGLE AND DIAGONAL
---------------------------
--upper corr writes the correlation (--method pearson, spearman or
kendall) in the upper cells, coloured blue (negative) to red (positive),
instead of repeating the scatter plots.  With -g it writes r over all rows
and r within each group: they can differ, even in sign (in iris, sepal
length and width give -0.12 overall and +0.5 to +0.75 within each species).
--diag hist draws histograms:

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
                    _corr_text(ax, df, groups, xcol, ycol, args.method, cmap, colors)
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

        # The column's range down each column of panels, and along each row
        limits = [_limits(df, col) for col in cols]
        for i in range(k):
            for j in range(k):
                if i < j and args.upper == "corr":
                    continue
                if limits[j] is not None:
                    axes[i][j].set_xlim(limits[j])
                if i != j and limits[i] is not None:
                    axes[i][j].set_ylim(limits[i])

        if args.groupcol is not None:
            # From a scatter panel: every group is there, with or without a KDE
            handles, labels = axes[1][0].get_legend_handles_labels()
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
