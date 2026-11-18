"""
dftk.commands.box_cmd — dftk box subcommand.

Box plot (or violin plot) of a value by category, with the individual
points jittered on top if wanted: e.g. protein level by loss-of-function
genotype (0/1/2), tips by day.  Drawn with seaborn, which is imported
only when the command runs.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io, option_columns
from dftk.common.plot import (
    WONG_PALETTE,
    add_figure_arguments,
    add_refline_arguments,
    apply_reflines,
    apply_style,
    make_figure,
    save_or_show,
)

logger = logging.getLogger(__name__)

_BOX_COLOUR = WONG_PALETTE[1]  # sky blue, without -g
_POINT_COLOUR = "0.15"

_EPILOG = """\
Input: one row per observation.  -y is the value, -x the category (one
box each, in sorted order unless --order gives one; numbers such as a
genotype 0/1/2 are categories too).  Without -x, a single box of all
rows.  Rows without a value or category are left out.

The box spans the quartiles with a line at the median, and the whiskers
reach the furthest point within 1.5 times the box's height (Tukey);
points beyond them are drawn on their own, unless --points draws every
point anyway.  --kind violin draws the distribution's density instead.

-g splits every box into one per group, side by side and in colour.

EXAMPLES
--------
Basic box plot of a value by category:

  dftk box data.tsv -x genotype -y protein -f box.png

Tips by day, with every bill drawn on top, and how many there are:

  dftk dataset tips -o | dftk box -x day -y tip --points --count \\
      --order Thur Fri Sat Sun -f tips.png

A reference line: the mean tip over all days is about 3 dollars:

  dftk dataset tips -o | dftk box -x day -y tip --hline 3:mean \\
      --order Thur Fri Sat Sun -f tips_mean.png

Smokers and non-smokers side by side within each day:

  dftk dataset tips -o | dftk box -x day -y tip -g smoker \\
      --order Thur Fri Sat Sun -f tips_smoker.png

Violins of body mass by species, split by sex:

  dftk dataset penguins -o | dftk box -x species -y body_mass_g -g sex \\
      --kind violin -f penguins.png

A level by genotype (0, 1 or 2 copies of a loss-of-function allele), on
a log axis: a made-up level that halves with each copy (diamond prices
as the spread, genotypes drawn with randvar):

  dftk dataset diamonds -o | dftk sample -n 2000 --randomseed 1 -o \\
      | dftk randvar -d u --dist uniform --randomseed 2 -o \\
      | dftk eval -f "dosage = (u > 0.6) * 1 + (u > 0.95) * 1" \\
          -f "level = price / 2**dosage" -o \\
      | dftk box -x dosage -y level --logy --points --count -f level.png
"""


def _order(values: pd.Series, given: list[str] | None) -> list:
    """Categories in sorted order (numbers numerically), or *given* first."""
    found = list(pd.unique(values.dropna()))
    try:
        found = sorted(found)
    except TypeError:
        found = sorted(found, key=str)
    if not given:
        return found
    by_text = {str(v): v for v in found}
    missing = [g for g in given if g not in by_text]
    if missing:
        raise ValueError(
            f"--order names values not in the data: {missing}; "
            f"found {[str(v) for v in found]}."
        )
    listed = [by_text[g] for g in given]
    return listed + [v for v in found if v not in listed]


class BoxCommand(BaseCommand):
    name = "box"
    help = "Box or violin plot of a value by category, points optional."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        c = parser.add_argument_group("columns")
        c.add_argument(
            "-y", "--ycol", required=True, metavar="COL", help="Value column."
        )
        c.add_argument(
            "-x",
            "--xcol",
            metavar="COL",
            help="Category column, one box each (default: one box of all rows).",
        )
        c.add_argument(
            "-g",
            "--groupcol",
            metavar="COL",
            help="Split each box into one per group, side by side, in colour.",
        )
        c.add_argument(
            "--order",
            nargs="+",
            metavar="VAL",
            help="Order of the -x categories (default: sorted); values not "
            "listed follow.",
        )
        c.add_argument(
            "--groupcolorder",
            nargs="+",
            metavar="VAL",
            help="Order of the -g groups (default: sorted).",
        )

        b = parser.add_argument_group("box")
        b.add_argument(
            "--kind",
            choices=["box", "violin"],
            default="box",
            help="box (quartiles and whiskers, the default) or violin (density).",
        )
        b.add_argument(
            "--points",
            action="store_true",
            help="Draw every point on top, jittered sideways.",
        )
        b.add_argument(
            "--count",
            action="store_true",
            help="Write the number of values under each category.",
        )
        b.add_argument(
            "--markersize",
            type=float,
            default=3.0,
            metavar="S",
            help="Size of the --points (default: 3).",
        )
        b.add_argument(
            "--alpha",
            type=float,
            default=0.5,
            metavar="A",
            help="Opacity of the --points, 0-1 (default: 0.5).",
        )

        a = parser.add_argument_group("axes")
        a.add_argument("-xl", "--xlabel", metavar="TEXT", help="X-axis label.")
        a.add_argument("-yl", "--ylabel", metavar="TEXT", help="Y-axis label.")
        a.add_argument(
            "-ym",
            "--ylim",
            nargs=2,
            type=float,
            metavar=("LO", "HI"),
            help="Y-axis limits.",
        )
        a.add_argument("-ly", "--logy", action="store_true", help="Log y axis.")
        a.add_argument(
            "--legendloc",
            nargs=2,
            type=float,
            metavar=("X", "Y"),
            help="Position of the -g legend, in axes coordinates (default: best).",
        )

        add_figure_arguments(parser)
        add_refline_arguments(parser, vertical=False)
        parser.set_defaults(size="7x4.5")

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args, columns=option_columns(args))
        check_cols(df, [args.ycol], "-y/--ycol")
        if args.xcol:
            check_cols(df, [args.xcol], "-x/--xcol")
        if args.groupcol:
            check_cols(df, [args.groupcol], "-g/--groupcol")
        if args.order and not args.xcol:
            raise ValueError("--order orders the -x categories; give -x.")
        if args.groupcolorder and not args.groupcol:
            raise ValueError("--groupcolorder orders the -g groups; give -g.")

        data = df.copy()
        values = pd.to_numeric(data[args.ycol], errors="coerce")
        bad = values.isna() & data[args.ycol].notna()
        if bad.any():
            raise ValueError(
                f"-y column {args.ycol!r} isn't numeric "
                f"(e.g. {data.loc[bad, args.ycol].iloc[0]!r})."
            )
        data[args.ycol] = values
        needed = [c for c in (args.ycol, args.xcol, args.groupcol) if c]
        keep = data[needed].notna().all(axis=1)
        if not keep.all():
            logger.info(
                "Left out %d of %d rows without %s",
                int((~keep).sum()),
                len(data),
                " / ".join(needed),
            )
        data = data[keep]
        if data.empty:
            raise ValueError("No rows with a value to plot.")
        if args.logy and (data[args.ycol] <= 0).any():
            raise ValueError(f"--logy needs -y values above 0 ({args.ycol!r}).")

        order = _order(data[args.xcol], args.order) if args.xcol else None
        hue_order = (
            _order(data[args.groupcol], args.groupcolorder) if args.groupcol else None
        )

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)
        fig, axes = make_figure(args)
        ax = axes[0][0]
        draw(ax, data, args, order, hue_order)
        ax.set_title(args.title or "")
        fig.tight_layout()
        save_or_show(fig, args)


def draw(ax, data: pd.DataFrame, args, order, hue_order) -> None:
    import seaborn as sns

    # matplotlib logs "Using categorical units ..." for numeric categories
    logging.getLogger("matplotlib.category").setLevel(logging.WARNING)

    common = dict(data=data, x=args.xcol, y=args.ycol, order=order, ax=ax)
    if args.groupcol:
        common.update(
            hue=args.groupcol,
            hue_order=hue_order,
            palette=WONG_PALETTE[: len(hue_order)],
        )
    else:
        common["color"] = _BOX_COLOUR
    if args.kind == "box":
        sns.boxplot(
            **common,
            showfliers=not args.points,
            width=0.6,
            linewidth=1,
            log_scale=args.logy or None,
        )
    else:
        sns.violinplot(
            **common,
            inner="quart" if not args.points else None,
            cut=0,
            linewidth=1,
            log_scale=args.logy or None,
        )
    if args.points:
        strip = dict(common)
        strip.pop("palette", None)
        strip.pop("hue_order", None)
        strip.pop("color", None)
        if args.groupcol:
            strip.update(
                hue_order=hue_order,
                dodge=True,
                palette=[_POINT_COLOUR] * len(hue_order),
            )
        sns.stripplot(
            **strip,
            color=_POINT_COLOUR if not args.groupcol else None,
            size=args.markersize,
            alpha=args.alpha,
            jitter=0.2,
            legend=False,
            zorder=3,
        )

    if args.count and args.xcol:
        counts = data.groupby(args.xcol, sort=False).size()
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(
            [
                f"{t.get_text()}\n(n={counts.get(v, 0)})"
                for t, v in zip(ax.get_xticklabels(), order, strict=True)
            ]
        )
    elif args.count:
        ax.set_xticks([0])
        ax.set_xticklabels([f"n={len(data)}"])

    if args.groupcol and ax.get_legend() is not None:
        ax.legend(
            title=args.groupcol,
            fontsize="small",
            title_fontsize="small",
            loc=tuple(args.legendloc) if args.legendloc else "best",
        )
    before = len(ax.texts)
    apply_reflines(ax, args)
    for label in ax.texts[before:]:
        # The line runs behind the boxes; its label stays readable over one
        label.set(zorder=5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
    if args.ylim is not None:
        ax.set_ylim(*args.ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(args.xlabel if args.xlabel is not None else (args.xcol or ""))
    ax.set_ylabel(args.ylabel if args.ylabel is not None else args.ycol)
