"""
dftk.commands.manhattan_cmd — dftk manhattan subcommand.

Genome-wide association plot: -log10(p) of every variant against its
position, chromosomes laid end to end in genome order.  Column lookup,
-log10(p), chromosome order and thinning live in common/gwas.py, shared
with the locus plot.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common import gwas
from dftk.common.io import check_cols, io
from dftk.common.plot import (
    WONG_PALETTE,
    add_figure_arguments,
    add_group_arguments,
    add_legend_arguments,
    apply_style,
    make_figure,
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

logger = logging.getLogger(__name__)

_CHROM_COLOURS = ("#1F4E79", "#7FA7CF")  # dark / light blue, alternating
_SIG_COLOUR = "#D55E00"  # vermilion (Wong)
_SUGGESTIVE_COLOUR = "#0072B2"  # blue (Wong)
_BAND_COLOUR = "#EDEDED"  # alternate chromosomes' background with -g

_EPILOG = """\
Input: one row per variant with a chromosome, a position and a p-value.
The columns are Chr, pos and Pval unless named; a missing default is looked
for under common spellings, ignoring case (CHR, #CHROM, BP, POS, P, …), and
the column used is logged.  Instead of a p-value, give -log10(p) with
--mlogpcol (regenie's LOG10P) or a chi-squared statistic with --chi2col
(--chi2df, default 1): p-values below about 1e-308 are stored as 0, and a
chi-squared column still gives their true -log10(p).

Chromosomes are drawn in genome order (1-22, X, Y, XY, MT, then others; a
"chr" prefix is ignored), each spanning its first to last variant.  Points
with p > 0.01 are thinned to one per cell of a grid finer than the figure
can show (--no-thin to draw all): a scan of ten million variants draws a
few hundred thousand points and looks the same.  Points are rasterized, so
a PDF stays small.  Rows without a chromosome, position or p-value are
left out.

EXAMPLES
--------
Basic Manhattan plot (columns Chr, pos, Pval):

  dftk manhattan data.tsv -f manhattan.png
  dftk dataset gwas -o | dftk manhattan -f manhattan.png

From a chi-squared column (exact even where Pval underflows to 0):

  dftk dataset gwas -o | dftk manhattan --chi2col Chisq -f manhattan.png

Other thresholds; no suggestive line:

  dftk dataset gwas -o | dftk manhattan --sig 1e-6 --suggestive none -f m.png

Cap the y axis at 40 so the smaller peaks show; points above it are drawn
as triangles on the top edge:

  dftk dataset gwas -o | dftk manhattan --ylim 0 40 -f manhattan.png

Several scans in one table (a study column): one panel each, or one colour
each with -g:

  dftk manhattan all_studies.tsv --subgraphcol study -f manhattan.png
  dftk manhattan all_studies.tsv -g study -f manhattan.png

Publication figure (Nature double column, PDF):

  dftk dataset gwas -o | dftk manhattan --size 7.2x2.6 --fontsize publication \\
      -f manhattan.pdf
"""


def _threshold(value: str) -> float | None:
    """--sig / --suggestive: a p-value, or 'none' for no line."""
    if value.lower() == "none":
        return None
    try:
        p = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a p-value or 'none', not {value!r}"
        ) from None
    if not 0 < p < 1:
        raise argparse.ArgumentTypeError(f"p-value must be in (0, 1), not {value!r}")
    return p


def add_gwas_column_arguments(parser: argparse.ArgumentParser) -> None:
    """Column options shared by manhattan and locus."""
    g = parser.add_argument_group("columns")
    g.add_argument(
        "--chrcol",
        metavar="COL",
        help=f"Chromosome column (default: {gwas.DEFAULT_CHR}).",
    )
    g.add_argument(
        "--poscol",
        metavar="COL",
        help=f"Position column (default: {gwas.DEFAULT_POS}).",
    )
    p = g.add_mutually_exclusive_group()
    p.add_argument(
        "--pcol",
        metavar="COL",
        help=f"P-value column (default: {gwas.DEFAULT_P}).",
    )
    p.add_argument(
        "--mlogpcol",
        metavar="COL",
        help="Column that already holds -log10(p), instead of a p-value.",
    )
    p.add_argument(
        "--chi2col",
        metavar="COL",
        help="Chi-squared column, instead of a p-value; exact for p below 1e-308.",
    )
    g.add_argument(
        "--chi2df",
        default="1",
        metavar="N|COL",
        help="Degrees of freedom of --chi2col: a number or a column (default: 1).",
    )


def gwas_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return df with _chrom (label), _pos and _y (-log10 p); rows lacking any dropped.

    The other columns are kept (for -g / --subgraphcol).
    """
    chrcol = gwas.find_column(
        df, args.chrcol, gwas.DEFAULT_CHR, gwas.CHR_ALIASES, "--chrcol"
    )
    poscol = gwas.find_column(
        df, args.poscol, gwas.DEFAULT_POS, gwas.POS_ALIASES, "--poscol"
    )
    y, _ = gwas.resolve_neglogp(df, args.pcol, args.mlogpcol, args.chi2col, args.chi2df)
    out = df.copy()
    out["_chrom"] = df[chrcol]
    out["_pos"] = pd.to_numeric(df[poscol], errors="coerce")
    out["_y"] = y
    keep = out["_chrom"].notna() & out["_pos"].notna() & np.isfinite(out["_y"])
    if not keep.all():
        logger.info(
            "Left out %d of %d rows without a chromosome, position or p-value",
            int((~keep).sum()),
            len(out),
        )
    out = out[keep]
    if out.empty:
        raise ValueError("No rows with a chromosome, a position and a p-value.")
    return out


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _ylimits(y: np.ndarray, lines: list[float], args) -> tuple[float, float]:
    top = max([float(np.max(y)) if y.size else 1.0, *lines, 1.0]) * 1.05
    lo, hi = 0.0, top
    if args.ylim is not None:
        lo = _float_or(args.ylim[0], lo)
        hi = _float_or(args.ylim[1], hi)
    return lo, hi


def _float_or(s, fallback: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return fallback


def _scatter(ax, x, y, hi, args, colours=None, **kw):
    """Points below *hi*, and triangles on the top edge for those above it.

    *colours*: one colour per point (alternating chromosomes), or None to
    use kw["color"].
    """
    over = y > hi
    size = args.markersize
    common = dict(alpha=args.alpha, linewidths=0, **kw)
    ax.scatter(
        x[~over], y[~over], s=size, rasterized=True,
        c=None if colours is None else colours[~over], **common,
    )  # fmt: skip
    if over.any():
        common.pop("label", None)
        ax.scatter(
            x[over], np.full(over.sum(), hi), s=size * 4, marker="^",
            clip_on=False, c=None if colours is None else colours[over], **common,
        )  # fmt: skip


def _draw(ax, data: pd.DataFrame, layout, args, groupcols=None) -> None:
    centres, labels, bounds = layout
    x = data["_x"].to_numpy()
    y = data["_y"].to_numpy()
    lines = [-np.log10(p) for p in (args.sig, args.suggestive) if p is not None]
    lo, hi = _ylimits(y, lines, args)

    groups = None
    if groupcols:
        groups = data[groupcols].astype(str).agg("\t".join, axis=1).to_numpy()
    keep = (
        gwas.thin_mask(x, y, groups=groups)
        if args.thin
        else np.ones(len(x), dtype=bool)
    )
    if args.thin and (~keep).any():
        logger.info("Thinned %d of %d points (p > 0.01)", int((~keep).sum()), len(x))

    if groupcols:
        # with colours used for groups, grey bands tell the chromosomes apart
        pad = _gap(bounds)
        for i, (start, end) in enumerate(bounds):
            if i % 2:
                ax.axvspan(start - pad, end + pad, color=_BAND_COLOUR, lw=0, zorder=0)
        for n, (name, gdf) in enumerate(
            subgraph_groups(data[keep], groupcols, args.groupcolorder)
        ):
            label = make_grouplabel(name, groupcols, args.groupcolformat)
            colour = WONG_PALETTE[n % len(WONG_PALETTE)]
            _scatter(
                ax, gdf["_x"].to_numpy(), gdf["_y"].to_numpy(), hi, args,
                color=colour, label=label, zorder=2,
            )  # fmt: skip
        ax.legend(
            loc=tuple(args.legendloc) if args.legendloc else "best",
            title=args.legendtitle,
            markerscale=3,
        )
    else:
        colours = np.array(args.colors)[data["_cidx"].to_numpy() % len(args.colors)]
        _scatter(
            ax, x[keep], y[keep], hi, args, colours=colours[keep], zorder=2,
            label=args.legend,
        )  # fmt: skip
        if args.legend:
            ax.legend(loc=tuple(args.legendloc) if args.legendloc else "best")

    if args.sig is not None:
        ax.axhline(-np.log10(args.sig), color=_SIG_COLOUR, lw=0.8, ls="--", zorder=3)
    if args.suggestive is not None:
        ax.axhline(
            -np.log10(args.suggestive), color=_SUGGESTIVE_COLOUR, lw=0.8, ls=":",
            zorder=3,
        )  # fmt: skip

    ax.set_ylim(lo, hi)
    pad = _gap(bounds)
    ax.set_xlim(bounds[0][0] - pad, bounds[-1][1] + pad)
    ax.set_xticks(centres)
    ax.set_xticklabels(_tick_labels(labels, ax))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.xlabel if args.xlabel is not None else "Chromosome")
    ax.set_ylabel(args.ylabel if args.ylabel is not None else r"$-\log_{10}(p)$")


def _gap(bounds) -> float:
    """Half the space between two chromosomes (or 0.5 % of the axis)."""
    if len(bounds) > 1:
        return (bounds[1][0] - bounds[0][1]) / 2
    return max(bounds[0][1] - bounds[0][0], 1.0) * 0.005


def _tick_labels(labels: list[str], ax) -> list[str]:
    """Every chromosome, or every other one when they'd crowd a narrow axes."""
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    if len(labels) <= 1.5 * width_in * 2:
        return labels
    return [c if i % 2 == 0 or not c.isdigit() else "" for i, c in enumerate(labels)]


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class ManhattanCommand(BaseCommand):
    name = "manhattan"
    help = "Manhattan plot: -log10(p) along the genome (Chr, pos, Pval)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_gwas_column_arguments(parser)

        g = parser.add_argument_group("manhattan")
        g.add_argument(
            "--sig",
            type=_threshold,
            default=5e-8,
            metavar="P|none",
            help="Genome-wide significance line, as a p-value (default: 5e-8).",
        )
        g.add_argument(
            "--suggestive",
            type=_threshold,
            default=1e-5,
            metavar="P|none",
            help="Suggestive line, as a p-value (default: 1e-5).",
        )
        g.add_argument(
            "--no-thin",
            dest="thin",
            action="store_false",
            help="Draw every point (default: thin points with p > 0.01).",
        )
        g.add_argument(
            "--colors",
            nargs="+",
            default=list(_CHROM_COLOURS),
            metavar="COLOUR",
            help="Colours that alternate along the chromosomes "
            "(default: dark and light blue).",
        )
        g.add_argument(
            "--markersize",
            type=float,
            default=4.0,
            metavar="S",
            help="Point area in pt² (default: 4).",
        )
        g.add_argument(
            "--alpha",
            type=float,
            default=1.0,
            metavar="A",
            help="Point opacity, 0-1 (default: 1).",
        )

        a = parser.add_argument_group("axes")
        a.add_argument("-xl", "--xlabel", metavar="TEXT", help="X-axis label.")
        a.add_argument("-yl", "--ylabel", metavar="TEXT", help="Y-axis label.")
        a.add_argument(
            "-ym",
            "--ylim",
            nargs=2,
            metavar=("LO", "HI"),
            help="Y-axis limits in -log10(p); points above HI are drawn as "
            "triangles on the top edge.",
        )

        add_figure_arguments(parser)
        parser.set_defaults(size="12x4")
        add_group_arguments(parser)
        add_legend_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        data = gwas_table(df, args)

        x, centres, labels, bounds = gwas.genome_positions(data["_chrom"], data["_pos"])
        data["_x"] = x
        data["_cidx"], _ = gwas.chrom_index(data["_chrom"])
        layout = (centres, labels, bounds)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)

        if args.subgraphcol is not None:
            groups = subgraph_groups(data, args.subgraphcol, args.subgraphorder)
            nrows, ncols = subgraph_layout(len(groups), args.ncols or 1)
            fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
            for idx, (name, gdf) in enumerate(groups):
                ax = axes[idx // ncols][idx % ncols]
                _draw(ax, gdf, layout, args, args.groupcol)
                ax.set_title(
                    make_grouplabel(name, args.subgraphcol, args.subgraphformat)
                )
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            if args.title:
                fig.suptitle(args.title)
        else:
            fig, axes = make_figure(args)
            _draw(axes[0][0], data, layout, args, args.groupcol)
            axes[0][0].set_title(args.title or "")
        fig.tight_layout()
        save_or_show(fig, args)
