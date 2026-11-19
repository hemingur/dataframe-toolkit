"""
dftk.commands.heatmap_cmd — dftk heatmap subcommand.

A coloured grid of values, e.g. a correlogram from ``dftk corr --all``.

Two input shapes:

  long  (-x COL -y COL -v COL)  one row per cell, as corr and stat give it;
                                 rows and columns keep their first-seen order
  wide  (-y COL)                 one row per heatmap row; every other numeric
                                 column (or --select) is a heatmap column,
                                 as pivot gives it or a matrix file holds it

--subgraphcol draws one heatmap per value of a column, side by side, with the
same rows, columns and colour scale in every panel and one colour bar.
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
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)


def _matrix(
    df: pd.DataFrame, args: argparse.Namespace, exclude: tuple[str, ...] = ()
) -> pd.DataFrame:
    """The heatmap as a DataFrame: index = y labels, columns = x labels.

    *exclude*: columns that are never heatmap columns in wide input
    (the --subgraphcol columns).
    """
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
            and c not in exclude
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


def _shown(matrix: pd.DataFrame, args: argparse.Namespace) -> np.ma.MaskedArray:
    """The cells drawn: missing values, and with --lower the upper triangle,
    masked."""
    values = matrix.to_numpy()
    shown = np.ma.masked_invalid(values)
    if args.lower:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("--lower needs a square heatmap (as corr --all gives)")
        shown = np.ma.masked_where(np.triu(np.ones_like(values), k=1) > 0, shown)
    return shown


def _scale(values: np.ndarray, args: argparse.Namespace) -> dict:
    """Colour map and limits from all the cells drawn (of every panel)."""
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
    elif finite.size:
        # Explicit limits, so that every panel shares them
        vmin = finite.min() if vmin is None else vmin
        vmax = finite.max() if vmax is None else vmax
    return {"cmap": cmap, "vmin": vmin, "vmax": vmax}


def _heatmap(
    ax,
    matrix: pd.DataFrame,
    shown: np.ma.MaskedArray,
    args: argparse.Namespace,
    scale: dict,
    fmt: str,
    xticklabels: bool = True,
    yticklabels: bool = True,
):
    """Draw one heatmap; return the mesh (for the colour bar)."""
    mesh = ax.pcolormesh(shown, **scale)

    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    if xticklabels:
        ax.set_xticklabels([str(c) for c in matrix.columns], rotation=45, ha="right")
        ax.set_xlabel(args.xlabel if args.xlabel is not None else (args.xcol or ""))
    else:
        ax.set_xticklabels([])
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    if yticklabels:
        ax.set_yticklabels([str(i) for i in matrix.index])
        ax.set_ylabel(args.ylabel if args.ylabel is not None else (args.ycol or ""))
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()  # first row at the top, as in the table
    ax.set_aspect("equal" if args.square else "auto")

    if args.annot:
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
    return mesh


def _panels(
    df: pd.DataFrame, args: argparse.Namespace
) -> list[tuple[str, pd.DataFrame]]:
    """[(title, matrix)]: one panel without --subgraphcol, else one per value,
    all with the rows and columns of the whole table in first-seen order."""
    if not args.subgraphcol:
        return [("", _matrix(df, args))]
    check_cols(df, args.subgraphcol, "--subgraphcol")
    clash = set(args.subgraphcol) & {args.xcol, args.ycol, args.value}
    if clash:
        raise ValueError(
            f"--subgraphcol {', '.join(sorted(clash))} is also -x, -y or -v"
        )
    exclude = tuple(args.subgraphcol)
    panels = [
        (
            make_grouplabel(key, args.subgraphcol, args.subgraphformat),
            _matrix(sub, args, exclude),
        )
        for key, sub in subgraph_groups(df, args.subgraphcol, args.subgraphorder)
    ]
    if not panels:
        raise ValueError(f"--subgraphcol {' '.join(args.subgraphcol)} has no values")
    # Every panel gets the rows and columns of the whole table, in the order
    # they first appear there, so that the panels line up cell for cell
    df = df.dropna(subset=args.subgraphcol)  # as the panels (groupby) do
    rows = pd.unique(df[args.ycol])
    cols = pd.unique(df[args.xcol]) if args.xcol else panels[0][1].columns
    return [(t, m.reindex(index=rows, columns=cols)) for t, m in panels]


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

PANELS
------
--subgraphcol draws one heatmap per value of a column, side by side
(--ncols per row, --subgraphorder, --subgraphformat for the titles). Every
panel has the rows and columns of the whole table, in the same places, and
the same colour scale, with one colour bar: a colour means the same value in
every panel. Mean body mass by species and island, one panel per sex:

  dftk dataset penguins -o \\
    | dftk pivot -i species island sex -v body_mass_g -f mean -o \\
    | dftk heatmap -x island -y species -v mean_body_mass_g \\
        --subgraphcol sex --annot --fmt .0f

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

        s = parser.add_argument_group("panels")
        s.add_argument(
            "--subgraphcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="One heatmap per value of this column, with the same rows, "
            "columns and colour scale.",
        )
        s.add_argument(
            "--subgraphorder",
            nargs="+",
            default=None,
            metavar="VAL",
            help="Panel order for --subgraphcol. Values not listed appear last "
            "in sorted order.",
        )
        s.add_argument(
            "--subgraphformat",
            default=None,
            metavar="TEMPLATE",
            help="Panel title template, with {colname} placeholders, e.g. "
            "'Genotype {genotype}'.",
        )
        s.add_argument(
            "--ncols",
            type=int,
            default=None,
            metavar="N",
            help="Panels per row (default: min(panels, 3)).",
        )

        add_figure_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        if args.select:
            check_cols(df, args.select, "--select")
            keep = [args.ycol] if args.ycol else []
            keep += [c for c in args.subgraphcol or [] if c not in keep]
            df = df[keep + [c for c in args.select if c not in keep]]
        panels = _panels(df, args)
        shown = [_shown(m, args) for _, m in panels]
        drawn = np.concatenate([s.filled(np.nan).ravel() for s in shown])
        scale = _scale(drawn, args)
        fmt = _fmt(drawn, args.fmt)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if getattr(args, "usetex", False):
            plt.rc("text", usetex=True)

        apply_style(args)
        if not args.subgraphcol:
            fig, axes = make_figure(args)
            ax = axes[0][0]
            mesh = _heatmap(ax, panels[0][1], shown[0], args, scale, fmt)
            ax.set_title(args.title or "")
            fig.colorbar(mesh, ax=ax, label=args.value or "")
            fig.tight_layout()
            save_or_show(fig, args)
            return

        nrows, ncols = subgraph_layout(len(panels), args.ncols)
        fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
        # Room for one shared colour bar; "compressed" also closes the gaps that
        # --square leaves between panels
        fig.set_layout_engine("compressed" if args.square else "constrained")
        used = []
        for idx, ((title, matrix), cells) in enumerate(zip(panels, shown, strict=True)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            # Tick labels on the outer edge only: the left column, and the
            # lowest panel of each column
            lowest = idx + ncols >= len(panels)
            mesh = _heatmap(
                ax, matrix, cells, args, scale, fmt,
                xticklabels=lowest, yticklabels=col == 0,
            )  # fmt: skip
            ax.set_title(title)
            if lowest and row < nrows - 1:
                # An empty grid cell below: its labels hang into that space,
                # so the layout needn't widen the gap under the whole row
                ax.xaxis.set_in_layout(False)
            used.append(ax)
        for idx in range(len(panels), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        fig.colorbar(mesh, ax=used, label=args.value or "")
        if args.title:
            fig.suptitle(args.title)
        save_or_show(fig, args)
