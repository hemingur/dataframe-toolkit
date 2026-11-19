"""
dftk.commands.phewas_cmd — dftk phewas subcommand.

PheWAS plot: -log10(p) of each phenotype, grouped by category along the x
axis (one colour per category), triangles pointing up or down with the
sign of the effect, a significance line (Bonferroni by default) and the
strongest hits labelled.  P-value column options and the log scale are
shared with manhattan.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.commands.manhattan_cmd import (
    add_gwas_column_arguments,
    float_or,
    log_yaxis,
    scatter_capped,
)
from dftk.common import gwas
from dftk.common.io import check_cols, io, option_columns
from dftk.common.plot import (
    WONG_PALETTE,
    add_figure_arguments,
    apply_style,
    make_figure,
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

logger = logging.getLogger(__name__)

DEFAULT_CATEGORY = "category"
DEFAULT_PHENOTYPE = "phenotype"
_SIG_COLOUR = "#D55E00"
_SUGGESTIVE_COLOUR = "#0072B2"
# Wong's palette without its black and yellow (unreadable as text on
# white), then darker tones for more categories
_CATEGORY_COLOURS = (
    *[c for c in WONG_PALETTE if c.lower() not in ("#000000", "#f0e442")],
    "#1F4E79",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#8C564B",
    "#80BA5A",
)
_GAP = 2.0  # x gap between categories, in phenotype widths

_EPILOG = """\
Input: one row per test (phenotype), with a category, a phenotype name
and a p-value (columns category, phenotype and Pval unless named; the
p-value as in manhattan: --pcol, --mlogpcol or --chi2col, common spellings
found ignoring case).  Categories are drawn left to right in order of
first appearance, each in its own colour, and the phenotypes in file
order within them: sort first to order them differently.

--betacol names a signed effect (beta, log odds ratio): points become
triangles pointing up where it is above 0 and down where it is below.
For odds ratios, take the log first (eval -f "logor = log(OR)").

The significance line is at 0.05 divided by the number of tests
(Bonferroni) unless --sig gives a p-value; --sig none removes it.  The
phenotypes above it are labelled, the strongest first, at most --label N
of them (default 10; 0 for none); labels that would overlap are moved
down.

One row per phenotype: several tests of a phenotype (e.g. heterozygous and
homozygous carriers) would stand side by side; pick one with query, or
draw one panel each with --subgraphcol.

--subgraphcol COL draws one PheWAS per value of COL (a burden model, a
gene, a genotype), stacked (--ncols for a grid).  A phenotype stands at the
same x in every panel, a missing one leaves a gap, and the panels share one
y axis.  The Bonferroni line counts each panel's own tests (it is logged
when the panels differ); --sig P draws one line at P in every panel.

EXAMPLES
--------
Basic PheWAS plot of a results table (category, phenotype, Pval, beta):

  dftk phewas results.tsv --betacol beta -f phewas.png

The simulated scan of heterozygous loss-of-function carriers:

  dftk dataset phewas -o | dftk query -q "genotype == 'het'" -o \\
      | dftk phewas --betacol beta -f phewas.png

Heterozygous and homozygous carriers, one panel each:

  dftk dataset phewas -o | dftk phewas --betacol beta \\
      --subgraphcol genotype -f phewas_by_genotype.png

Fixed threshold, a suggestive line, more room for the labels:

  dftk dataset phewas -o | dftk query -q "genotype == 'het'" -o \\
      | dftk phewas --betacol beta --sig 1e-5 --suggestive 1e-3 \\
          --size 10x5 -f phewas.png
"""


def _sig_threshold(value: str) -> float | str | None:
    """--sig: a p-value, 'bonferroni' or 'none'."""
    if value.lower() in ("bonferroni", "none"):
        return None if value.lower() == "none" else "bonferroni"
    return _p_or_none(value)


def _p_or_none(value: str) -> float | None:
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


def phewas_table(df: pd.DataFrame, args) -> pd.DataFrame:
    """df with _cat, _label, _y (-log10 p) and _sign (+1, -1, 0); rows
    without a category or p-value dropped."""
    out = df.copy()
    out["_cat"] = df[args.catcol]
    out["_label"] = df[args.labelcol].astype(str)
    y, _ = gwas.resolve_neglogp(df, args.pcol, args.mlogpcol, args.chi2col, args.chi2df)
    out["_y"] = y
    if args.betacol:
        beta = pd.to_numeric(df[args.betacol], errors="coerce")
        bad = beta.isna() & df[args.betacol].notna()
        if bad.any():
            raise ValueError(
                f"--betacol column {args.betacol!r} isn't numeric "
                f"(e.g. {df.loc[bad, args.betacol].iloc[0]!r})."
            )
        out["_sign"] = np.sign(beta.fillna(0)).astype(int)
    else:
        out["_sign"] = 0
    keep = out["_cat"].notna() & np.isfinite(out["_y"])
    if not keep.all():
        logger.info(
            "Left out %d of %d rows without a category or p-value",
            int((~keep).sum()),
            len(out),
        )
    out = out[keep]
    if out.empty:
        raise ValueError("No rows with a category and a p-value.")
    out["_cat"] = out["_cat"].astype(str)
    return out


def layout(data: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    """x of every row, and (category, first x, last x) of each category:
    categories in order of first appearance, rows in file order within."""
    cats = list(dict.fromkeys(data["_cat"]))
    x = np.empty(len(data))
    spans = []
    start = 0.0
    codes = data["_cat"].to_numpy()
    for cat in cats:
        idx = np.flatnonzero(codes == cat)
        x[idx] = start + np.arange(len(idx))
        spans.append((cat, start, start + len(idx) - 1))
        start += len(idx) - 1 + _GAP
    return x, spans


def shared_layout(
    panels: list[pd.DataFrame],
) -> tuple[list[np.ndarray], list[tuple[str, float, float]]]:
    """x of every row of every panel, and the category spans: one x per
    phenotype (category and label) across all panels, so that a phenotype
    stands at the same place in each; one missing from a panel leaves a gap."""
    keys = pd.concat([p[["_cat", "_label"]] for p in panels]).drop_duplicates()
    x, spans = layout(keys.reset_index(drop=True))
    at = dict(zip(zip(keys["_cat"], keys["_label"], strict=True), x, strict=True))
    xs = [
        np.array([at[k] for k in zip(p["_cat"], p["_label"], strict=True)], dtype=float)
        for p in panels
    ]
    return xs, spans


def significance(args, n: int) -> float | None:
    if args.sig == "bonferroni":
        return 0.05 / n
    return args.sig


def _lines(args, n: int) -> list[float]:
    """-log10 of the significance and suggestive lines for n tests."""
    sig = significance(args, n)
    return [-np.log10(p) for p in (sig, args.suggestive) if p is not None]


def _ylim(panels: list[pd.DataFrame], args) -> tuple[float, float]:
    """y limits shared by all panels: room for the highest point or line."""
    top = max(max([float(p["_y"].max()), *_lines(args, len(p))]) for p in panels)
    lo, hi = 0.0, max(top, 1.0) * 1.08
    if args.ylim is not None:
        lo, hi = float_or(args.ylim[0], lo), float_or(args.ylim[1], hi)
    return lo, hi


def _draw(
    ax,
    data: pd.DataFrame,
    args,
    x: np.ndarray,
    spans: list[tuple[str, float, float]],
    ylim: tuple[float, float],
    legend: bool = True,
    categories: bool = True,
    ylabel: bool = True,
) -> None:
    y = data["_y"].to_numpy()
    sign = data["_sign"].to_numpy()
    sig = significance(args, len(data))
    lo, hi = ylim

    colour_of = {
        cat: _CATEGORY_COLOURS[i % len(_CATEGORY_COLOURS)]
        for i, (cat, _, _) in enumerate(spans)
    }
    colours = np.array([colour_of[c] for c in data["_cat"]], dtype=object)
    for s, marker in ((1, "^"), (-1, "v"), (0, "o")):
        m = sign == s
        if m.any():
            scatter_capped(
                ax,
                x[m],
                y[m],
                hi,
                args,
                colours=colours[m],
                marker=marker,
                capmarker="^",
                zorder=2,
            )

    if sig is not None:
        ax.axhline(-np.log10(sig), color=_SIG_COLOUR, lw=0.8, ls="--", zorder=1)
    if args.suggestive is not None:
        ax.axhline(
            -np.log10(args.suggestive),
            color=_SUGGESTIVE_COLOUR,
            lw=0.8,
            ls=":",
            zorder=1,
        )

    if sig is not None and args.label > 0:
        above = np.flatnonzero(y >= -np.log10(sig))
        top = above[np.argsort(-y[above], kind="stable")][: args.label]
        right = x.max() - 0.15 * (x.max() - x.min())
        for i in top:
            near_edge = x[i] > right  # label to the left, inside the axes
            ax.annotate(
                data["_label"].iloc[i],
                (x[i], min(y[i], hi)),
                xytext=(-4 if near_edge else 4, 4),
                textcoords="offset points",
                ha="right" if near_edge else "left",
                va="bottom",
                fontsize="small",
                zorder=5,
                annotation_clip=False,
            )
        if len(above) > len(top):
            logger.info(
                "Labelled the %d strongest of %d phenotypes above the line "
                "(--label N for more)",
                len(top),
                len(above),
            )

    if legend and args.betacol and (sign != 0).any():
        from matplotlib.lines import Line2D

        handles = [
            Line2D([], [], ls="", marker=m, color="0.3", markersize=6, label=t)
            for m, t in (
                ("^", f"{args.betacol} > 0"),
                ("v", f"{args.betacol} < 0"),
            )
        ]
        ax.legend(
            handles=handles,
            fontsize="small",
            loc=tuple(args.legendloc) if args.legendloc else "best",
        )

    ax.set_xticks([(a + b) / 2 for _, a, b in spans])
    if categories:
        ax.set_xticklabels([c for c, _, _ in spans], rotation=45, ha="right")
        for label, (cat, _, _) in zip(ax.get_xticklabels(), spans, strict=True):
            label.set_color(colour_of[cat])
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-_GAP / 2 - 0.5, spans[-1][2] + _GAP / 2 + 0.5)
    ax.set_ylim(lo, hi)
    if args.logy is not None:
        log_yaxis(ax, args.logy, lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.xlabel if args.xlabel is not None and categories else "")
    default_ylabel = r"$-\log_{10}(p)$"
    ax.set_ylabel(
        (args.ylabel if args.ylabel is not None else default_ylabel) if ylabel else ""
    )


def spread_labels(fig, ax) -> None:
    """Move labels down until they don't overlap the ones placed before
    them (the strongest hits are placed first)."""
    renderer = fig.canvas.get_renderer()
    placed = []
    for text in ax.texts:
        for _ in range(40):
            box = text.get_window_extent(renderer)
            if not any(box.overlaps(p) for p in placed):
                break
            dx, dy = text.xyann
            text.xyann = (dx, dy - box.height * 72 / fig.dpi * 0.5)
        placed.append(text.get_window_extent(renderer))


def _panels(data: pd.DataFrame, args) -> tuple[list[str], list[pd.DataFrame]]:
    """Titles and rows of the --subgraphcol panels; logs each panel's
    Bonferroni threshold when they differ."""
    check_cols(data, args.subgraphcol, "--subgraphcol")
    clash = set(args.subgraphcol) & {args.catcol, args.labelcol}
    if clash:
        raise ValueError(f"--subgraphcol {', '.join(sorted(clash))} is also -c or -l")
    groups = subgraph_groups(data, args.subgraphcol, args.subgraphorder)
    if not groups:
        raise ValueError(f"--subgraphcol {' '.join(args.subgraphcol)} has no values")
    titles = [
        make_grouplabel(k, args.subgraphcol, args.subgraphformat) for k, _ in groups
    ]
    panels = [g.reset_index(drop=True) for _, g in groups]
    counts = [len(p) for p in panels]
    if args.sig == "bonferroni" and len(set(counts)) > 1:
        logger.info(
            "Bonferroni line per panel: %s",
            "; ".join(
                f"{t}: 0.05/{n} = {0.05 / n:.2g}"
                for t, n in zip(titles, counts, strict=True)
            ),
        )
    return titles, panels


class PhewasCommand(BaseCommand):
    name = "phewas"
    help = "PheWAS plot: -log10(p) of each phenotype, grouped by category."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        c = add_gwas_column_arguments(parser, position=False)
        c.add_argument(
            "-c",
            "--catcol",
            default=DEFAULT_CATEGORY,
            metavar="COL",
            help=f"Category column (default: {DEFAULT_CATEGORY}).",
        )
        c.add_argument(
            "-l",
            "--labelcol",
            default=DEFAULT_PHENOTYPE,
            metavar="COL",
            help=f"Phenotype name column, for the labels "
            f"(default: {DEFAULT_PHENOTYPE}).",
        )
        c.add_argument(
            "--betacol",
            metavar="COL",
            help="Signed effect (beta, log OR): triangles up above 0, down below.",
        )

        g = parser.add_argument_group("phewas")
        g.add_argument(
            "--sig",
            type=_sig_threshold,
            default="bonferroni",
            metavar="P|bonferroni|none",
            help="Significance line (default: bonferroni, 0.05 / number of tests).",
        )
        g.add_argument(
            "--suggestive",
            type=_p_or_none,
            default=None,
            metavar="P|none",
            help="Suggestive line, as a p-value (default: none).",
        )
        g.add_argument(
            "--label",
            type=int,
            default=10,
            metavar="N",
            help="Label at most N phenotypes above the significance line, "
            "strongest first (default: 10; 0 for none).",
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
            "--markersize",
            type=float,
            default=30.0,
            metavar="S",
            help="Point area in pt² (default: 30).",
        )
        g.add_argument(
            "--alpha",
            type=float,
            default=0.9,
            metavar="A",
            help="Point opacity, 0-1 (default: 0.9).",
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
            help="Position of the --betacol legend, in axes coordinates "
            "(default: best).",
        )

        s = parser.add_argument_group("panels")
        s.add_argument(
            "--subgraphcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="One PheWAS per value of this column (a burden model, a gene), "
            "stacked, with the phenotypes at the same x and one y axis; the "
            "Bonferroni line counts each panel's own tests.",
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
            "'Model: {model}'.",
        )
        s.add_argument(
            "--ncols",
            type=int,
            default=1,
            metavar="N",
            help="Panels per row (default: 1, stacked).",
        )

        add_figure_arguments(parser)
        parser.set_defaults(size="8x4.5")

    def execute(self, args: argparse.Namespace) -> None:
        if args.logy is not None and args.logy <= 0:
            raise ValueError(f"--logy must be above 0, not {args.logy:g}.")
        if args.label < 0:
            raise ValueError(f"--label must be 0 or more, not {args.label}.")
        df = io.read(args, columns=_columns(args))
        check_cols(df, [args.catcol], "-c/--catcol")
        check_cols(df, [args.labelcol], "-l/--labelcol")
        if args.betacol:
            check_cols(df, [args.betacol], "--betacol")
        data = phewas_table(df, args).reset_index(drop=True)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)
        if not args.subgraphcol:
            fig, axes = make_figure(args)
            ax = axes[0][0]
            x, spans = layout(data)
            _draw(ax, data, args, x, spans, _ylim([data], args))
            ax.set_title(args.title or "")
            fig.tight_layout()
            spread_labels(fig, ax)
            save_or_show(fig, args)
            return

        titles, panels = _panels(data, args)
        xs, spans = shared_layout(panels)
        ylim = _ylim(panels, args)
        nrows, ncols = subgraph_layout(len(panels), args.ncols)
        fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
        used = []
        for idx, (title, panel, x) in enumerate(zip(titles, panels, xs, strict=True)):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            # Category names under the lowest panel of each column only
            lowest = idx + ncols >= len(panels)
            _draw(
                ax, panel, args, x, spans, ylim,
                legend=idx == 0, categories=lowest, ylabel=col == 0,
            )  # fmt: skip
            ax.set_title(title)
            if lowest and row < nrows - 1:
                # An empty grid cell below: its labels hang into that space
                ax.xaxis.set_in_layout(False)
            used.append(ax)
        for idx in range(len(panels), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)
        if args.title:
            fig.suptitle(args.title)
        fig.tight_layout()
        for ax in used:
            spread_labels(fig, ax)
        save_or_show(fig, args)


def _columns(args: argparse.Namespace):
    """columns= for io.read: the p-value column (named or found by alias)
    and the ones options name (category and phenotype by default)."""
    named = option_columns(args)

    def pick(names: list[str]) -> list[str] | None:
        found = gwas.needed_columns(names, args, position=False)
        return None if found is None else list(dict.fromkeys(found + named(names)))

    return pick
