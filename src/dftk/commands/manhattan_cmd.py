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
from dftk.common.io import check_cols, io, option_columns
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

Or log-scale it above -log10(p) = 10 (linear below), so a peak at 380
and one at 8 both show:

  dftk dataset gwas -o | dftk manhattan --chi2col Chisq --logy -f manhattan.png

Several scans in one table (a study column): one panel each, or one colour
each with -g:

  dftk manhattan all_studies.tsv --subgraphcol study -f manhattan.png
  dftk manhattan all_studies.tsv -g study -f manhattan.png

Two scans face to face (a Miami plot): the first value of the study column
above the axis, the second mirrored below it; points past --sig in red.
Here the second scan is the example scan with every chi-squared halved,
and melt stacks the two columns into one, with a study column:

  dftk dataset gwas -o \
      | dftk eval -f "half = Chisq / 2" --select Chr pos Chisq half -o \
      | dftk melt -i Chr pos -d study -v chi2 -o \
      | dftk manhattan --chi2col chi2 --miami study --sigcolor red -f miami.png

With more than two scans in the column, name the two: --miami study a b.

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


def add_gwas_column_arguments(
    parser: argparse.ArgumentParser, position: bool = True
) -> None:
    """Column options shared by manhattan and qq (qq: position=False)."""
    g = parser.add_argument_group("columns")
    if position:
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


def gwas_columns(args: argparse.Namespace, position: bool = True):
    """columns= for io.read: the chromosome, position and p-value columns
    (found as gwas_table will find them) and the ones options name."""
    named = option_columns(args)

    def pick(names: list[str]) -> list[str] | None:
        found = gwas.needed_columns(names, args, position)
        return None if found is None else found + named(names)

    return pick


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


def _ylimits(
    y: np.ndarray, lines: list[float], args, top: np.ndarray | None = None
) -> tuple[float, float]:
    """Axis limits; with *top* (Miami), the lower half's limit is negative."""

    def _peak(v):
        return max([float(np.max(v)) if v.size else 1.0, *lines, 1.0]) * 1.05

    if top is None:
        lo, hi = 0.0, _peak(y)
    else:
        lo, hi = -_peak(-y[~top]), _peak(y[top])
    if args.ylim is not None:
        lo = float_or(args.ylim[0], lo)
        hi = float_or(args.ylim[1], hi)
    return lo, hi


def float_or(s, fallback: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return fallback


def scatter_capped(ax, x, y, hi, args, colours=None, lo=None, **kw):
    """Points within the axis, and triangles on its edge for those beyond it.

    Points above *hi* go on the top edge (pointing up); with *lo* (a Miami
    plot's lower half), points below it go on the bottom edge (pointing
    down).  *colours*: one colour per point (alternating chromosomes), or
    None to use kw["color"].
    """
    over = y > hi
    under = y < lo if lo is not None else np.zeros(len(y), dtype=bool)
    inside = ~over & ~under
    size = args.markersize
    common = dict(alpha=args.alpha, linewidths=0, **kw)
    ax.scatter(
        x[inside], y[inside], s=size, rasterized=True,
        c=None if colours is None else colours[inside], **common,
    )  # fmt: skip
    common.pop("label", None)
    for mask, edge, marker in ((over, hi, "^"), (under, lo, "v")):
        if mask.any():
            ax.scatter(
                x[mask], np.full(mask.sum(), edge), s=size * 4, marker=marker,
                clip_on=False, c=None if colours is None else colours[mask],
                **common,
            )  # fmt: skip


def log_yaxis(ax, linthresh: float, lo: float, hi: float) -> None:
    """Symmetric-log y axis: linear up to *linthresh*, log10 above it.

    Ticks: a few on the linear part, then 1, 2, 5 times each power of ten
    (only the powers when that would crowd the axis); the same below zero
    for a Miami plot.  Labels as plain numbers without signs.
    """
    from matplotlib.ticker import MaxNLocator, NullLocator

    ax.set_yscale("symlog", linthresh=linthresh, linscale=1.0)
    top = max(hi, -lo)
    lin = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, linthresh)
    pos = {float(t) for t in lin if 0 <= t <= linthresh}
    k0 = int(np.floor(np.log10(linthresh)))
    k1 = int(np.ceil(np.log10(max(top, linthresh))))
    for mults in ((1, 2, 5), (1,)):
        log = {
            m * 10.0**k
            for k in range(k0, k1 + 1)
            for m in mults
            if linthresh < m * 10.0**k <= top
        }
        if len(log) <= 6:
            break
    pos |= log
    ticks = sorted(pos | {-t for t in pos})
    ax.set_yticks([t for t in ticks if lo <= t <= hi])
    ax.yaxis.set_minor_locator(NullLocator())
    _abs_ticks(ax)


def _abs_ticks(ax) -> None:
    """Y tick labels without the minus sign (a Miami plot's lower half)."""
    from matplotlib.ticker import FuncFormatter

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{abs(v):g}"))


def _draw(ax, data: pd.DataFrame, layout, args, groupcols=None) -> None:
    centres, labels, bounds = layout
    x = data["_x"].to_numpy()
    y = data["_y"].to_numpy()
    lines = [-np.log10(p) for p in (args.sig, args.suggestive) if p is not None]

    # Miami: the second trait is drawn below the axis, at -(-log10 p)
    top = None
    if args.miami:
        top = (data["_half"] == 0).to_numpy()
        y = np.where(top, y, -y)
    lo, hi = _ylimits(y, lines, args, top)

    groups = None
    if groupcols:
        groups = data[groupcols].astype(str).agg("\t".join, axis=1).to_numpy()
    elif top is not None:
        groups = top
    keep = (
        gwas.thin_mask(x, np.abs(y), groups=groups)
        if args.thin
        else np.ones(len(x), dtype=bool)
    )
    if args.thin and (~keep).any():
        logger.info("Thinned %d of %d points (p > 0.01)", int((~keep).sum()), len(x))
    floor = lo if top is not None else None

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
            scatter_capped(
                ax, gdf["_x"].to_numpy(), gdf["_y"].to_numpy(), hi, args,
                color=colour, label=label, zorder=2,
            )  # fmt: skip
        ax.legend(
            loc=tuple(args.legendloc) if args.legendloc else "best",
            title=args.legendtitle,
            markerscale=3,
        )
    else:
        colours = np.array(args.colors, dtype=object)[
            data["_cidx"].to_numpy() % len(args.colors)
        ]
        if args.sigcolor and args.sig is not None:
            colours[np.abs(y) >= -np.log10(args.sig)] = args.sigcolor
        scatter_capped(
            ax, x[keep], y[keep], hi, args, colours=colours[keep], lo=floor,
            zorder=2, label=args.legend,
        )  # fmt: skip
        if args.legend:
            ax.legend(loc=tuple(args.legendloc) if args.legendloc else "best")

    signs = (1, -1) if top is not None else (1,)
    for p, colour, style in (
        (args.sig, _SIG_COLOUR, "--"),
        (args.suggestive, _SUGGESTIVE_COLOUR, ":"),
    ):
        if p is not None:
            for sign in signs:
                ax.axhline(
                    sign * -np.log10(p), color=colour, lw=0.8, ls=style, zorder=3
                )

    ax.set_ylim(lo, hi)
    if args.logy is not None:
        log_yaxis(ax, args.logy, lo, hi)
    if top is not None:
        ax.axhline(0, color="black", lw=0.6, zorder=3)
        _abs_ticks(ax)
        upper, lower = args.miami_names
        ax.text(0.01, 0.98, upper, transform=ax.transAxes, ha="left", va="top")
        ax.text(0.01, 0.02, lower, transform=ax.transAxes, ha="left", va="bottom")
    pad = _gap(bounds)
    ax.set_xlim(bounds[0][0] - pad, bounds[-1][1] + pad)
    ax.set_xticks(centres)
    ax.set_xticklabels(_tick_labels(labels, ax))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.xlabel if args.xlabel is not None else "Chromosome")
    ax.set_ylabel(args.ylabel if args.ylabel is not None else r"$-\log_{10}(p)$")


def miami_halves(data: pd.DataFrame, spec: list[str]) -> tuple[pd.DataFrame, list]:
    """Add _half (0 above the axis, 1 below) from --miami COL [UPPER LOWER].

    With COL alone it must hold exactly two values, the first one in the
    table drawn above; with UPPER and LOWER, rows with other values are
    left out.
    """
    col, *names = spec
    check_cols(data, [col], "--miami")
    if len(names) not in (0, 2):
        raise ValueError(
            "--miami takes a column, or a column and the two values to draw "
            f"(above, below), not {len(spec)} arguments."
        )
    values = data[col].astype(str)
    if names:
        found = set(values)
        missing = [n for n in names if n not in found]
        if missing:
            raise ValueError(
                f"--miami: {missing} not in column {col!r}; it holds "
                f"{sorted(found)[:10]}"
            )
        if names[0] == names[1]:
            raise ValueError("--miami: the two values must differ.")
    else:
        names = list(pd.unique(values))
        if len(names) != 2:
            raise ValueError(
                f"--miami: column {col!r} holds {len(names)} values, not 2; "
                f"name the two to draw: --miami {col} UPPER LOWER"
            )
    keep = values.isin(names).to_numpy()
    if not keep.all():
        logger.info("--miami: left out %d rows of other values", int((~keep).sum()))
    out = data[keep].copy()
    out["_half"] = (values[keep] == names[1]).astype(int).to_numpy()
    return out, names


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
            "--miami",
            nargs="+",
            metavar="COL",
            help="Two scans in one plot, the second mirrored below the axis "
            "(a Miami plot): --miami COL [UPPER LOWER].  COL alone must hold two "
            "values, the first drawn above; or name the two to draw.",
        )
        g.add_argument(
            "--logy",
            nargs="?",
            type=float,
            const=10.0,
            metavar="T",
            help="Log scale above -log10(p) = T, linear below it (default T: 10), "
            "so a peak of 300 and one of 10 both show.",
        )
        g.add_argument(
            "--no-thin",
            dest="thin",
            action="store_false",
            help="Draw every point (default: thin points with p > 0.01).",
        )
        g.add_argument(
            "--sigcolor",
            metavar="COLOUR",
            help="Colour for points past the --sig line (default: the "
            "chromosome colours; not with -g).",
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
            "triangles on the top edge.  With --miami the lower half is "
            "negative: --ylim -30 30.",
        )

        add_figure_arguments(parser)
        parser.set_defaults(size=None)
        add_group_arguments(parser)
        add_legend_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args, columns=gwas_columns(args, position=True))
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        if args.miami and args.groupcol:
            raise ValueError("--miami and -g/--groupcol can't be combined.")
        if args.logy is not None and args.logy <= 0:
            raise ValueError(f"--logy must be above 0, not {args.logy:g}.")
        if args.size is None:
            args.size = "12x6" if args.miami else "12x4"
        data = gwas_table(df, args)
        if args.miami:
            data, args.miami_names = miami_halves(data, args.miami)

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
