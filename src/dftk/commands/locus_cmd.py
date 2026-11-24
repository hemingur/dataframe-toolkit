"""
dftk.commands.locus_cmd — dftk locus subcommand.

Regional association plot, the zoomed-in companion to manhattan: -log10(p)
of the variants in one region against position (Mb), coloured by LD (r²)
with a lead variant drawn as a diamond.  Column lookup, -log10(p), capped
points, thinning and the log scale are shared with manhattan.
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.commands.manhattan_cmd import (
    _threshold,
    add_gwas_column_arguments,
    float_or,
    gwas_table,
    log_yaxis,
    scatter_capped,
    thin_points,
)
from dftk.common import gwas
from dftk.common.io import check_cols, io, option_columns
from dftk.common.plot import (
    add_figure_arguments,
    apply_style,
    make_figure,
    save_or_show,
)

logger = logging.getLogger(__name__)

DEFAULT_ID = "SNP"
ID_ALIASES = (
    "snp", "id", "rsid", "rs_id", "snpid", "variant_id", "marker", "markername",
)  # fmt: skip

# r² bins: 0-0.2, 0.2-0.4, ..., 0.8-1 (dark blue to red), grey when missing
_LD_EDGES = (0.2, 0.4, 0.6, 0.8)
_LD_COLOURS = ("#357EBD", "#46B8DA", "#5CB85C", "#EEA236", "#D43F3A")
_NO_LD_COLOUR = "#BBBBBB"
_POINT_COLOUR = "#1F4E79"  # without --ldcol: manhattan's dark blue
_LEAD_COLOUR = "#9632B8"  # purple diamond
_SIG_COLOUR = "#D55E00"
_SUGGESTIVE_COLOUR = "#0072B2"

_POSITION = re.compile(r"^(?:chr)?([0-9A-Za-z]+):([0-9][0-9,_]*)$", re.IGNORECASE)

_EPILOG = """\
Input: one row per variant on one chromosome, with a position and a
p-value, as in manhattan (columns Chr, pos and Pval unless named; common
spellings found ignoring case; --mlogpcol or --chi2col instead of a
p-value).  The plot covers the rows it is given: cut the region out with
query first.  Input with several chromosomes is an error.

The lead variant is drawn as a purple diamond and labelled; it is the
smallest p-value unless --lead names one.  --label writes the names of
other variants next to their points.  A variant is named by position,
chr6:32600000 (or 6:32600000), or by ID in the ID column (SNP unless
--idcol names it; rsid, ID, ... found ignoring case).  Labels show the ID
when there is an ID column, the position otherwise.  Several rows at one
position (multi-allelic sites): the one with the smallest p.

--ldcol colours the points by r² with the lead variant, in five bins
(0-0.2, ..., 0.8-1), grey where it is missing.  dftk doesn't compute LD:
r² comes from elsewhere (e.g. plink --r2 --ld-snp rs123 --ld-window-kb
1000) and is merged in.  It must be r² with the variant --lead names;
dftk can't check that.

EXAMPLES
--------
Basic locus plot of a region file (Chr, pos, Pval; r2 with the lead):

  dftk locus region.tsv --ldcol r2 -f locus.png

The chr 6 peak of the example scan (its r2 column is r² with the peak's
top variant), with the region cut out by query:

  dftk dataset gwas -o \\
      | dftk query -q "Chr == '6' and pos > 31.6e6 and pos < 33.6e6" -o \\
      | dftk locus --ldcol r2 -f locus.png

Choose the lead and label a second signal (an ID or a position):

  dftk dataset gwas -o \\
      | dftk query -q "Chr == '6' and pos > 31.6e6 and pos < 33.6e6" -o \\
      | dftk locus --ldcol r2 --lead sim43271 --label chr6:32436692 -f locus.png

Log scale above -log10(p) = 10, so the peak and the weaker signals both
show:

  dftk dataset gwas -o \\
      | dftk query -q "Chr == '6' and pos > 31.6e6 and pos < 33.6e6" -o \\
      | dftk locus --chi2col Chisq --ldcol r2 --logy -f locus.png
"""


def id_column(df: pd.DataFrame, given: str | None) -> tuple[str | None, str | None]:
    """(ID column, None), or (None, why not) when there isn't one to use.

    Looked up as the other columns are, but only an error when a variant
    has to be found by ID (lookup_variant)."""
    try:
        return gwas.find_column(df, given, DEFAULT_ID, ID_ALIASES, "--idcol"), None
    except ValueError as e:
        if given is not None:
            raise
        return None, str(e)


def lookup_variant(
    data: pd.DataFrame, name: str, idcol: str | None, why: str | None
) -> int:
    """Index label in *data* of the variant *name*: an ID, or chr:pos.

    An ID match wins, so IDs like 6:32600000:A:G still work by ID.
    Several matches: the smallest p (largest _y).
    """
    rows = None
    if idcol is not None:
        hit = data[data[idcol].astype(str) == name]
        if len(hit):
            rows = hit
    if rows is None:
        m = _POSITION.match(name.strip())
        if m:
            chrom = gwas.chrom_label(m.group(1))
            pos = int(m.group(2).replace(",", "").replace("_", ""))
            rows = data[
                (data["_chrom"].map(gwas.chrom_label) == chrom) & (data["_pos"] == pos)
            ]
            if rows.empty:
                raise ValueError(f"No variant at {name} (with a p-value).")
        elif idcol is None:
            raise ValueError(
                f"{name!r} isn't a position like chr6:32600000, and there is no "
                f"ID column to look it up in: {why}"
            )
        else:
            raise ValueError(
                f"No variant {name!r} in column {idcol!r} (with a p-value)."
            )
    if len(rows) > 1:
        logger.warning(
            "%d rows match %s; using the one with the smallest p", len(rows), name
        )
    return rows["_y"].idxmax()


def _ld_colours(r2: np.ndarray) -> np.ndarray:
    colours = np.array(_LD_COLOURS, dtype=object)[np.digitize(r2, _LD_EDGES)]
    colours[~np.isfinite(r2)] = _NO_LD_COLOUR
    return colours


def _label_text(data: pd.DataFrame, row, idcol: str | None) -> str:
    if idcol is not None and pd.notna(data.at[row, idcol]):
        return str(data.at[row, idcol])
    return f"{gwas.chrom_label(data.at[row, '_chrom'])}:{int(data.at[row, '_pos'])}"


def _draw(ax, data: pd.DataFrame, lead, labels: list, args, idcol) -> None:
    x = data["_pos"].to_numpy(dtype=float) / 1e6
    y = data["_y"].to_numpy()
    lines = [-np.log10(p) for p in (args.sig, args.suggestive) if p is not None]
    lo, hi = 0.0, max([float(y.max()), *lines, 1.0]) * 1.08
    if args.ylim is not None:
        lo, hi = float_or(args.ylim[0], lo), float_or(args.ylim[1], hi)

    if args.ldcol:
        r2 = pd.to_numeric(data[args.ldcol], errors="coerce").to_numpy(dtype=float)
        colours = _ld_colours(r2)
        # the most LD on top: grey first, then by r²
        order = np.argsort(np.where(np.isfinite(r2), r2, -1.0), kind="stable")
    else:
        colours = np.full(len(x), _POINT_COLOUR, dtype=object)
        order = np.arange(len(x))

    keep = thin_points(x, y, args)
    marked = data.index.isin([lead, *labels])
    keep |= marked  # the lead and labelled variants are always drawn
    if args.thin and (~keep).any():
        logger.info("Thinned %d of %d points (p > 0.01)", int((~keep).sum()), len(x))
    order = order[keep[order]]
    scatter_capped(ax, x[order], y[order], hi, args, colours=colours[order], zorder=2)

    # lead: a diamond over its point
    i = data.index.get_loc(lead)
    ax.scatter(
        [x[i]], [min(y[i], hi)], s=args.markersize * 6, marker="D",
        color=_LEAD_COLOUR, edgecolors="black", linewidths=0.6, zorder=4,
        clip_on=False,
    )  # fmt: skip
    for row in dict.fromkeys([lead, *labels]):
        j = data.index.get_loc(row)
        if row != lead:  # a ring shows which point the label belongs to
            ax.scatter(
                [x[j]], [min(y[j], hi)], s=args.markersize * 3, facecolors="none",
                edgecolors="black", linewidths=0.8, zorder=3, clip_on=False,
            )  # fmt: skip
        ax.annotate(
            _label_text(data, row, idcol), (x[j], min(y[j], hi)),
            xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
            fontsize="small", zorder=5, annotation_clip=False,
        )  # fmt: skip

    if args.sig is not None:
        ax.axhline(-np.log10(args.sig), color=_SIG_COLOUR, lw=0.8, ls="--", zorder=1)
    if args.suggestive is not None:
        ax.axhline(
            -np.log10(args.suggestive), color=_SUGGESTIVE_COLOUR, lw=0.8, ls=":",
            zorder=1,
        )  # fmt: skip

    if args.ldcol:
        from matplotlib.lines import Line2D

        names = ["0.8–1", "0.6–0.8", "0.4–0.6", "0.2–0.4", "0–0.2", "no r²"]
        handles = [
            Line2D([], [], ls="", marker="o", markersize=5, color=c, label=n)
            for c, n in zip((*_LD_COLOURS[::-1], _NO_LD_COLOUR), names, strict=True)
        ]
        ax.legend(
            handles=handles, title="r²", fontsize="small", title_fontsize="small",
            loc=tuple(args.legendloc) if args.legendloc else "upper right",
        )  # fmt: skip

    span = max(x.max() - x.min(), 1e-3)
    ax.set_xlim(x.min() - 0.02 * span, x.max() + 0.02 * span)
    ax.set_ylim(lo, hi)
    if args.logy is not None:
        log_yaxis(ax, args.logy, lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    chrom = gwas.chrom_label(data["_chrom"].iloc[0])
    ax.set_xlabel(
        args.xlabel if args.xlabel is not None else f"Position on chr{chrom} (Mb)"
    )
    ax.set_ylabel(args.ylabel if args.ylabel is not None else r"$-\log_{10}(p)$")


class LocusCommand(BaseCommand):
    name = "locus"
    help = "Regional association plot: -log10(p) in one region, coloured by LD (r²)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        c = add_gwas_column_arguments(parser)
        c.add_argument(
            "--idcol",
            metavar="COL",
            help=f"Variant ID column, for --lead / --label by ID and for the "
            f"labels (default: {DEFAULT_ID}, if there is one).",
        )
        c.add_argument(
            "--ldcol",
            metavar="COL",
            help="r² with the lead variant: colours the points in five bins, "
            "grey where missing.",
        )

        g = parser.add_argument_group("locus")
        g.add_argument(
            "--lead",
            metavar="VARIANT",
            help="Lead variant, drawn as a diamond: an ID or chr:pos "
            "(default: the smallest p-value).",
        )
        g.add_argument(
            "--label",
            nargs="+",
            default=[],
            metavar="VARIANT",
            help="Variants to label (IDs or chr:pos); the lead is always labelled.",
        )
        g.add_argument(
            "--sig",
            type=_threshold,
            default=5e-8,
            metavar="P|none",
            help="Significance line, as a p-value (default: 5e-8).",
        )
        g.add_argument(
            "--suggestive",
            type=_threshold,
            default=None,
            metavar="P|none",
            help="Suggestive line, as a p-value (default: none).",
        )
        g.add_argument(
            "--logy",
            nargs="?",
            type=float,
            const=10.0,
            metavar="T",
            help="Log scale above -log10(p) = T, linear below it (default T: 10).",
        )
        g.add_argument(
            "--no-thin",
            dest="thin",
            action="store_false",
            help="Draw every point (default: thin points with p > 0.01).",
        )
        g.add_argument(
            "--markersize",
            type=float,
            default=12.0,
            metavar="S",
            help="Point area in pt² (default: 12).",
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
        a.add_argument(
            "--legendloc",
            nargs=2,
            type=float,
            metavar=("X", "Y"),
            help="Position of the r² legend, in axes coordinates "
            "(default: upper right).",
        )

        add_figure_arguments(parser)
        parser.set_defaults(default_size="7x4")

    def execute(self, args: argparse.Namespace) -> None:
        if args.logy is not None and args.logy <= 0:
            raise ValueError(f"--logy must be above 0, not {args.logy:g}.")
        df = io.read(args, columns=_columns(args))
        if args.ldcol:
            check_cols(df, [args.ldcol], "--ldcol")
        idcol, why = id_column(df, args.idcol)
        data = gwas_table(df, args)

        chroms = pd.unique(data["_chrom"].map(gwas.chrom_label))
        if len(chroms) > 1:
            raise ValueError(_one_region_hint(df, data, args, len(chroms)))

        data = data.reset_index(drop=True)
        if args.lead is not None:
            lead = lookup_variant(data, args.lead, idcol, why)
        else:
            lead = data["_y"].idxmax()
        labels = [lookup_variant(data, n, idcol, why) for n in args.label]
        logger.info("Lead variant: %s", _label_text(data, lead, idcol))

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)
        fig, axes = make_figure(args)
        _draw(axes[0][0], data, lead, labels, args, idcol)
        axes[0][0].set_title(args.title or "")
        fig.tight_layout()
        save_or_show(fig, args)


def _columns(args: argparse.Namespace):
    """columns= for io.read: the chromosome, position and p-value columns,
    the ID column (named or found by alias) and the ones options name."""
    named = option_columns(args)

    def pick(names: list[str]) -> list[str] | None:
        found = gwas.needed_columns(names, args, position=True)
        if found is None:
            return None
        frame = pd.DataFrame(columns=names)
        quiet, gwas.logger.disabled = gwas.logger.disabled, True
        try:
            idcol, _ = id_column(frame, args.idcol)
        except ValueError:
            idcol = None  # a missing --idcol is reported on the real read
        finally:
            gwas.logger.disabled = quiet
        extra = [idcol] if idcol is not None else []
        return list(dict.fromkeys(found + extra + named(names)))

    return pick


def _one_region_hint(df: pd.DataFrame, data: pd.DataFrame, args, n: int) -> str:
    """The error for input on several chromosomes: a query for 500 kb
    either side of the smallest p-value."""
    chrcol = gwas.find_column(
        df, args.chrcol, gwas.DEFAULT_CHR, gwas.CHR_ALIASES, "--chrcol"
    )
    poscol = gwas.find_column(
        df, args.poscol, gwas.DEFAULT_POS, gwas.POS_ALIASES, "--poscol"
    )
    top = data.loc[data["_y"].idxmax()]
    chrom = top["_chrom"]
    chrom = f"'{chrom}'" if isinstance(chrom, str) else f"{chrom}"
    pos = int(top["_pos"])
    lo, hi = max(pos - 500_000, 0), pos + 500_000
    return (
        f"locus draws one region, but the input has {n} chromosomes.  Cut "
        f"the region out first; 500 kb either side of the smallest p-value:\n"
        f'  dftk query -q "{chrcol} == {chrom} and {poscol} >= {lo} and '
        f'{poscol} <= {hi}" -o | dftk locus ...'
    )
