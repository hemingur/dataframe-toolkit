"""
dftk.commands.heatmap_cmd — dftk heatmap subcommand.

A coloured grid of values, e.g. a correlogram from ``dftk corr --all``.

Two input shapes:

  long  (-x COL -y COL -v COL)  one row per cell, as corr and stat give it;
                                 rows and columns keep their first-seen order
  wide  (-y COL)                 one row per heatmap row; every other numeric
                                 column (or --select) is a heatmap column,
                                 as pivot gives it or a matrix file holds it
"""

import argparse

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io
from dftk.common.plot import (
    add_figure_arguments,
    apply_style,
    make_figure,
    save_or_show,
)


def _matrix(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """The heatmap as a DataFrame: index = y labels, columns = x labels."""
    if args.xcol is None and args.value is None:
        if args.ycol is None:
            raise ValueError(
                "give -x, -y and -v (one row per cell), or -y alone (one row "
                "per heatmap row, other numeric columns as heatmap columns)"
            )
        check_cols(df, [args.ycol], "-y/--ycol")
        values = [
            c
            for c in df.columns
            if c != args.ycol
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]
        if not values:
            raise ValueError(f"no numeric columns besides -y {args.ycol}")
        if df[args.ycol].duplicated().any():
            raise ValueError(f"-y {args.ycol} has repeated values: one row each")
        return df.set_index(args.ycol)[values].astype(float)

    if None in (args.xcol, args.ycol, args.value):
        raise ValueError("long input needs all of -x, -y and -v")
    check_cols(df, [args.xcol, args.ycol, args.value], "-x/-y/-v")
    keys = [args.ycol, args.xcol]
    if df.duplicated(keys).any():
        raise ValueError(
            f"several rows for the same {args.xcol} and {args.ycol}: summarise "
            "first (dftk pivot), or filter (--prequery)"
        )
    xs = pd.unique(df[args.xcol])
    ys = pd.unique(df[args.ycol])
    matrix = df.pivot(index=args.ycol, columns=args.xcol, values=args.value)
    return matrix.reindex(index=ys, columns=xs).astype(float)


def _fmt(values: np.ndarray, fmt: str | None) -> str:
    """--fmt, or g for whole numbers (counts) and .2f otherwise."""
    if fmt:
        return fmt
    finite = values[np.isfinite(values)]
    return "g" if finite.size and np.all(finite == np.round(finite)) else ".2f"


def _heatmap(ax, fig, matrix: pd.DataFrame, args: argparse.Namespace) -> None:
    values = matrix.to_numpy()
    shown = np.ma.masked_invalid(values)
    if args.lower:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("--lower needs a square heatmap (as corr --all gives)")
        shown = np.ma.masked_where(np.triu(np.ones_like(values), k=1) > 0, shown)

    vmin, vmax, center = args.vmin, args.vmax, args.center
    finite = values[np.isfinite(values)]
    if center is None and finite.size and finite.min() < 0 < finite.max():
        center = 0.0  # values of both signs: diverging colours around 0
    cmap = args.cmap or ("RdBu_r" if center is not None else "viridis")
    if center is not None and finite.size:
        # Symmetric limits so that center gets the middle colour
        half = max(abs(finite.max() - center), abs(finite.min() - center))
        vmin = center - half if vmin is None else vmin
        vmax = center + half if vmax is None else vmax

    mesh = ax.pcolormesh(shown, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(mesh, ax=ax, label=args.value or "")

    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_yticklabels([str(i) for i in matrix.index])
    ax.invert_yaxis()  # first row at the top, as in the table
    ax.set_xlabel(args.xlabel if args.xlabel is not None else (args.xcol or ""))
    ax.set_ylabel(args.ylabel if args.ylabel is not None else (args.ycol or ""))
    ax.set_title(args.title or "")
    ax.set_aspect("equal" if args.square else "auto")

    if args.annot:
        fmt = _fmt(values, args.fmt)
        for (i, j), v in np.ndenumerate(shown.filled(np.nan)):
            if np.isfinite(v):
                # Dark text on light cells, light text on dark cells
                r, g, b, _ = mesh.cmap(mesh.norm(v))
                light = 0.299 * r + 0.587 * g + 0.114 * b > 0.5
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    format(v, fmt),
                    ha="center",
                    va="center",
                    color="black" if light else "white",
                )


_EPILOG = """\
INPUT
-----
Long, one row per cell (-x, -y, -v): rows and columns of the heatmap keep
the order in which their labels first appear.

  dftk heatmap data.tsv -x sample -y gene -v expression
  dftk dataset tips -o | dftk pivot -i day time -v tip -f mean -o \\
    | dftk heatmap -x time -y day -v mean_tip --annot

Wide, one row per heatmap row (-y alone): every other numeric column is a
heatmap column (--select picks and orders them):

  dftk dataset tips -o | dftk pivot -i day -g time -v tip -f mean -o \\
    | dftk heatmap -y day --annot

CORRELOGRAM
-----------
corr --all gives every pair of columns both ways round, which is a square
heatmap. Values of both signs get diverging colours centred on 0.  Give
--vmin -1 --vmax 1 for correlations: otherwise the scale stretches to the
largest |r|, and a weak correlation looks as strong as a strong one would:

  dftk dataset iris -o | dftk corr --all -o \\
    | dftk heatmap -x col1 -y col2 -v correlation --annot --vmin -1 --vmax 1

  --lower keeps the lower triangle and the diagonal; --square makes the
  cells square:

  dftk dataset diamonds -o | dftk corr --all --method spearman -o \\
    | dftk heatmap -x col1 -y col2 -v correlation --annot --lower --square \\
        --vmin -1 --vmax 1

COLOURS AND NUMBERS
-------------------
--cmap takes any matplotlib colour map (viridis, magma, RdBu_r, coolwarm,
...). --annot writes each value in its cell, with --fmt as a Python format
(.2f, .1%, g; default g for whole numbers, else .2f). Empty cells stay
blank.

  dftk dataset penguins -o | dftk pivot -i species -g island -v body_mass_g \\
      -f count -o \\
    | dftk heatmap -y species --annot --cmap Blues -f counts.png
"""


class HeatmapCommand(BaseCommand):
    name = "heatmap"
    help = "Coloured grid of values, e.g. a correlogram from corr --all."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)

        g = parser.add_argument_group("heatmap")
        g.add_argument("-x", "--xcol", metavar="COL", help="Heatmap column labels.")
        g.add_argument(
            "-y",
            "--ycol",
            metavar="COL",
            help="Heatmap row labels.  Alone, without -x and -v: the other "
            "numeric columns are the heatmap columns.",
        )
        g.add_argument("-v", "--value", metavar="COL", help="Cell values.")
        g.add_argument(
            "--select",
            nargs="+",
            default=None,
            metavar="COL",
            help="Wide input: the value columns to show, in this order.",
        )
        g.add_argument("--xlabel", default=None, help="X-axis label (default: -x).")
        g.add_argument("--ylabel", default=None, help="Y-axis label (default: -y).")
        g.add_argument("--annot", action="store_true", help="Write values in cells.")
        g.add_argument(
            "--fmt",
            default=None,
            metavar="FMT",
            help="Format of --annot values, e.g. .2f or .1%% (default: g for "
            "whole numbers, else .2f).",
        )
        g.add_argument(
            "--cmap",
            default=None,
            help="Matplotlib colour map (default: RdBu_r around --center, "
            "else viridis).",
        )
        g.add_argument(
            "--center",
            type=float,
            default=None,
            help="Value that gets the middle colour (default: 0 when values "
            "have both signs).",
        )
        g.add_argument("--vmin", type=float, default=None, help="Colour scale minimum.")
        g.add_argument("--vmax", type=float, default=None, help="Colour scale maximum.")
        g.add_argument(
            "--lower",
            action="store_true",
            help="Show only the lower triangle and diagonal (square heatmaps).",
        )
        g.add_argument("--square", action="store_true", help="Square cells.")

        add_figure_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        if args.select:
            check_cols(df, args.select, "--select")
            keep = [args.ycol] if args.ycol else []
            df = df[keep + [c for c in args.select if c not in keep]]
        matrix = _matrix(df, args)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if getattr(args, "usetex", False):
            plt.rc("text", usetex=True)

        apply_style(args)
        fig, axes = make_figure(args)
        _heatmap(axes[0][0], fig, matrix, args)
        fig.tight_layout()
        save_or_show(fig, args)
