"""
dftk.commands.bar_cmd — dftk bar subcommand.

Bar chart of values already in the table, one bar per row: a mean per
cohort from `dftk stat -g`, a rate per gene.  Error bars from a column
(--yerr) or a pair of interval columns (--yci), groups side by side (-g),
panels (--subgraphcol), horizontal bars for long names.  Counts are not
computed here: summarise first (`dftk stat`, `dftk pivot`).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
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
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

logger = logging.getLogger(__name__)

_BAR_COLOUR = WONG_PALETTE[1]  # sky blue, without -g
_ERR_COLOUR = "0.15"
_GROUP_WIDTH = 0.8  # of one category slot, shared by its -g bars

_EPILOG = """\
Input: one row per bar, with the value already computed (a mean, a rate,
an estimate). -x is the category, -y the value. With -g, one row per
category and group; the groups stand side by side within each category.
Two rows for the same bar are an error: summarise first (dftk stat -g,
dftk pivot). Categories and groups keep the order in which they first
appear, unless --order / --groupcolorder give one. A group missing from a
category leaves a gap, so the bars stay aligned.

Bars start at 0, so negative values point down, and there is no log axis.

EXAMPLES
--------
Basic bar chart of a value per category:

  dftk bar data.tsv -x cohort -y rate -f bar.png

Mean tip per day, with the standard error of the mean:

  dftk dataset tips -o | dftk stat -g day -c tip -o \\
      | dftk bar -x day -y mean --yerr sem --order Thur Fri Sat Sun \\
          -yl "mean tip" -f tips.png

Mean body mass by species and sex, side by side, with the values written
on the bars:

  dftk dataset penguins -o | dftk stat -g species sex -c body_mass_g -o \\
      | dftk bar -x species -y mean --yerr sem -g sex --labels --fmt .0f \\
          -yl "body mass (g)" -f penguins.png

One panel per island, bars along the y axis (long names read better):

  dftk dataset penguins -o \\
      | dftk stat -g island species sex -c flipper_length_mm -o \\
      | dftk bar -x species -y mean --yerr sem -g sex \\
          --subgraphcol island --horizontal -f flippers.png

A reference value: --hline (--vline with --horizontal):

  dftk dataset tips -o | dftk stat -g day -c tip -o \\
      | dftk bar -x day -y mean --hline 3:overall -f tips_ref.png
"""


def ordered(values: pd.Series, given: list[str] | None, option: str) -> list:
    """Distinct values in order of first appearance, or *given* first."""
    found = list(pd.unique(values.dropna()))
    if not given:
        return found
    by_text = {str(v): v for v in found}
    missing = [g for g in given if g not in by_text]
    if missing:
        raise ValueError(
            f"{option} names values not in the data: {missing}; "
            f"found {[str(v) for v in found]}."
        )
    listed = [by_text[g] for g in given]
    return listed + [v for v in found if v not in listed]


def yci_columns(args) -> list[str]:
    """The two column names of --yci LO,HI ([] without it)."""
    if not args.yci:
        return []
    cols = [c.strip() for c in args.yci.split(",")]
    if len(cols) != 2 or not all(cols):
        raise ValueError(
            "--yci takes two column names separated by a comma, e.g. "
            f"--yci cilo,cihi; not {args.yci!r}."
        )
    return cols


def _numeric(data: pd.DataFrame, col: str, option: str) -> pd.Series:
    values = pd.to_numeric(data[col], errors="coerce")
    bad = values.isna() & data[col].notna()
    if bad.any():
        raise ValueError(
            f"{option} column {col!r} isn't numeric "
            f"(e.g. {data.loc[bad, col].iloc[0]!r})."
        )
    return values


def prepare(df: pd.DataFrame, args) -> pd.DataFrame:
    """Check the columns, keep rows with a bar, and reject duplicate bars.
    Adds _lo/_hi: the error bar's extent below and above the value."""
    check_cols(df, [args.xcol], "-x/--xcol")
    check_cols(df, [args.ycol], "-y/--ycol")
    if args.groupcol:
        check_cols(df, [args.groupcol], "-g/--groupcol")
    if args.yerr:
        check_cols(df, [args.yerr], "--yerr")
    ci = yci_columns(args)
    if ci:
        check_cols(df, ci, "--yci")
    panels = args.subgraphcol or []
    if panels:
        check_cols(df, panels, "--subgraphcol")
    if args.groupcolorder and not args.groupcol:
        raise ValueError("--groupcolorder orders the -g groups; give -g.")
    if args.subgraphorder and not panels:
        raise ValueError("--subgraphorder orders the panels; give --subgraphcol.")
    keys = [*panels, args.xcol] + ([args.groupcol] if args.groupcol else [])
    if len(set(keys)) < len(keys):
        raise ValueError("-x, -g and --subgraphcol must be different columns.")

    data = df.copy()
    data[args.ycol] = _numeric(data, args.ycol, "-y")
    keep = data[[*keys, args.ycol]].notna().all(axis=1)
    if not keep.all():
        logger.info(
            "Left out %d of %d rows without %s",
            int((~keep).sum()),
            len(data),
            " / ".join([*keys, args.ycol]),
        )
    data = data[keep].reset_index(drop=True)
    if data.empty:
        raise ValueError("No rows with a value to plot.")

    dup = data.duplicated(subset=keys, keep=False)
    if dup.any():
        example = data.loc[dup, keys].iloc[0].to_dict()
        raise ValueError(
            f"{int(dup.sum())} rows share a bar ({', '.join(keys)}), e.g. "
            f"{example}: one row per bar. Summarise first, e.g. dftk stat -g "
            f"{' '.join(keys)} -c {args.ycol}, then -y mean."
        )

    if args.yerr:
        err = _numeric(data, args.yerr, "--yerr").abs()
        data["_lo"], data["_hi"] = err, err
    elif ci:
        lo = _numeric(data, ci[0], "--yci")
        hi = _numeric(data, ci[1], "--yci")
        data["_lo"] = data[args.ycol] - lo
        data["_hi"] = hi - data[args.ycol]
        if ((data["_lo"] < 0) | (data["_hi"] < 0)).any():
            raise ValueError(
                f"--yci {args.yci}: the interval must contain the value "
                f"({ci[0]} <= {args.ycol} <= {ci[1]})."
            )
    return data


def bars(data: pd.DataFrame, args, categories: list, groups: list | None):
    """Rows of (position, value, lo, hi, group index) for one panel."""
    slot = {c: i for i, c in enumerate(categories)}
    n = len(groups) if groups else 1
    width = _GROUP_WIDTH / n
    gindex = {g: j for j, g in enumerate(groups)} if groups else None
    out = []
    for _, row in data.iterrows():
        j = gindex[row[args.groupcol]] if groups else 0
        pos = slot[row[args.xcol]] + (j - (n - 1) / 2) * width
        lo = row["_lo"] if "_lo" in row else np.nan
        hi = row["_hi"] if "_hi" in row else np.nan
        out.append((pos, row[args.ycol], lo, hi, j))
    return out, width


def draw(ax, data, args, categories, groups) -> None:
    """Draw one panel's bars, error bars and labels."""
    rows, width = bars(data, args, categories, groups)
    horizontal = args.horizontal
    palette = (WONG_PALETTE * (len(groups) // len(WONG_PALETTE) + 1)) if groups else []
    for j in range(len(groups) if groups else 1):
        mine = [r for r in rows if r[4] == j]
        if not mine and not groups:
            continue
        pos = [r[0] for r in mine]
        val = [r[1] for r in mine]
        colour = palette[j] if groups else _BAR_COLOUR
        label = str(groups[j]) if groups else None
        if horizontal:
            ax.barh(pos, val, height=width, color=colour, label=label, zorder=2)
        else:
            ax.bar(pos, val, width=width, color=colour, label=label, zorder=2)

    has_err = "_lo" in data
    if has_err:
        pos = np.array([r[0] for r in rows])
        val = np.array([r[1] for r in rows], dtype=float)
        err = np.array([[r[2] for r in rows], [r[3] for r in rows]], dtype=float)
        ok = ~np.isnan(err).any(axis=0)
        kw = dict(fmt="none", ecolor=_ERR_COLOUR, elinewidth=1, capsize=3, zorder=3)
        if horizontal:
            ax.errorbar(val[ok], pos[ok], xerr=err[:, ok], **kw)
        else:
            ax.errorbar(pos[ok], val[ok], yerr=err[:, ok], **kw)

    if args.labels:
        for pos, val, lo, hi, _ in rows:
            up = val >= 0
            reach = val + ((hi if up else -lo) if has_err and not np.isnan(lo) else 0)
            text = format(val, args.fmt)
            if horizontal:
                ax.annotate(
                    text, (reach, pos), xytext=(3 if up else -3, 0),
                    textcoords="offset points", va="center",
                    ha="left" if up else "right", fontsize="small", zorder=4,
                )  # fmt: skip
            else:
                ax.annotate(
                    text, (pos, reach), xytext=(0, 2 if up else -2),
                    textcoords="offset points", ha="center",
                    va="bottom" if up else "top", fontsize="small", zorder=4,
                )  # fmt: skip

    names = [str(c) for c in categories]
    ticks = range(len(categories))
    if horizontal:
        ax.set_yticks(ticks, names)
        ax.set_ylim(len(categories) - 0.5, -0.5)  # first category on top
        ax.axvline(0, color="0.3", lw=0.8, zorder=2.5)
        ax.grid(True, axis="x", linewidth=0.4, alpha=0.5, zorder=0)
    else:
        long = sum(len(n) for n in names) > 50 or max(map(len, names)) > 12
        ax.set_xticks(
            ticks, names, rotation=45 if long else 0, ha="right" if long else "center"
        )
        ax.set_xlim(-0.5, len(categories) - 0.5)
        ax.axhline(0, color="0.3", lw=0.8, zorder=2.5)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    before = len(ax.texts)
    apply_reflines(ax, args)
    for label in ax.texts[before:]:
        label.set(zorder=5, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def _value_limits(axes, args, labelled: bool) -> tuple[float, float]:
    """One value range for every panel, so the bars compare across panels."""
    if args.ylim is not None:
        return tuple(args.ylim)
    get = (lambda ax: ax.get_xlim()) if args.horizontal else (lambda ax: ax.get_ylim())
    lo = min(get(ax)[0] for ax in axes)
    hi = max(get(ax)[1] for ax in axes)
    if labelled:  # room for the numbers past the longest bar
        span = hi - lo
        hi += 0.06 * span if hi > 0 else 0
        lo -= 0.06 * span if lo < 0 else 0
    return lo, hi


class BarCommand(BaseCommand):
    name = "bar"
    help = "Bar chart of values per category, error bars and groups optional."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        c = parser.add_argument_group("columns")
        c.add_argument(
            "-x",
            "--xcol",
            required=True,
            metavar="COL",
            help="Category column: one bar (or one -g set of bars) each.",
        )
        c.add_argument(
            "-y",
            "--ycol",
            required=True,
            metavar="COL",
            help="Value column: the height of the bar.",
        )
        c.add_argument(
            "-g",
            "--groupcol",
            metavar="COL",
            help="Groups side by side within each category, in colour.",
        )
        c.add_argument(
            "--order",
            nargs="+",
            metavar="VAL",
            help="Order of the -x categories (default: as they first appear); "
            "values not listed follow.",
        )
        c.add_argument(
            "--groupcolorder",
            nargs="+",
            metavar="VAL",
            help="Order of the -g groups (default: as they first appear).",
        )

        e = parser.add_argument_group("error bars (one of)")
        mx = e.add_mutually_exclusive_group()
        mx.add_argument(
            "--yerr", metavar="COL", help="Column of symmetric ± error values."
        )
        mx.add_argument(
            "--yci",
            metavar="CILO,CIHI",
            help="Two columns with the interval's ends (comma-separated, "
            "e.g. cilo,cihi).",
        )

        b = parser.add_argument_group("bars")
        b.add_argument(
            "--horizontal",
            action="store_true",
            help="Bars along the x axis, categories top to bottom (for long names).",
        )
        b.add_argument(
            "--labels",
            action="store_true",
            help="Write each bar's value at its end (past the error bar).",
        )
        b.add_argument(
            "--fmt",
            default=".3g",
            metavar="SPEC",
            help="Number format of --labels (default: .3g; e.g. .1f, .0f).",
        )

        p = parser.add_argument_group("panels")
        p.add_argument(
            "--subgraphcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="One panel per value of this column, with the same "
            "categories, groups and value axis.",
        )
        p.add_argument(
            "--subgraphorder",
            nargs="+",
            default=None,
            metavar="VAL",
            help="Panel order for --subgraphcol. Values not listed appear last.",
        )
        p.add_argument(
            "--subgraphformat",
            default=None,
            metavar="TEMPLATE",
            help="Panel title template, with {colname} placeholders, e.g. "
            "'Model {model}'.",
        )
        p.add_argument(
            "--ncols",
            type=int,
            default=None,
            metavar="N",
            help="Panels per row (default: min(panels, 3)).",
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
            help="Limits of the value axis (the x axis with --horizontal).",
        )
        a.add_argument(
            "--legendloc",
            nargs=2,
            type=float,
            metavar=("X", "Y"),
            help="Position of the -g legend, in axes coordinates (default: best).",
        )

        add_figure_arguments(parser)
        add_refline_arguments(parser)
        parser.set_defaults(default_size="7x4.5")

    def execute(self, args: argparse.Namespace) -> None:
        if args.horizontal and args.hline:
            raise ValueError(
                "With --horizontal the values are on the x axis: use --vline."
            )
        if not args.horizontal and args.vline:
            raise ValueError(
                "The values are on the y axis: use --hline (--vline with --horizontal)."
            )
        df = io.read(args, columns=option_columns(args, *yci_columns(args)))
        data = prepare(df, args)
        categories = ordered(data[args.xcol], args.order, "--order")
        groups = (
            ordered(data[args.groupcol], args.groupcolorder, "--groupcolorder")
            if args.groupcol
            else None
        )
        if args.subgraphcol:
            panels = [
                (make_grouplabel(key, args.subgraphcol, args.subgraphformat), sub)
                for key, sub in subgraph_groups(
                    data, args.subgraphcol, args.subgraphorder
                )
            ]
        else:
            panels = [(args.title or "", data)]

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)
        nrows, ncols = (
            subgraph_layout(len(panels), args.ncols) if args.subgraphcol else (1, 1)
        )
        fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
        used = []
        for idx, (title, sub) in enumerate(panels):
            ax = axes[idx // ncols][idx % ncols]
            draw(ax, sub, args, categories, groups)
            ax.set_title(title)
            used.append(ax)
        for idx in range(len(panels), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        lo, hi = _value_limits(used, args, args.labels)
        value_label = args.ylabel if args.ylabel is not None else args.ycol
        cat_label = args.xlabel if args.xlabel is not None else args.xcol
        for idx, ax in enumerate(used):
            first_col = idx % ncols == 0
            lowest = idx + ncols >= len(panels)
            # the panels share their categories: name them on the outer
            # panels only (left column, or bottom row)
            if args.horizontal:
                ax.set_xlim(lo, hi)
                ax.set_xlabel(value_label if lowest else "")
                ax.set_ylabel(cat_label if first_col else "")
                ax.tick_params(axis="y", labelleft=first_col)
            else:
                ax.set_ylim(lo, hi)
                ax.set_ylabel(value_label if first_col else "")
                ax.set_xlabel(cat_label if lowest else "")
                ax.tick_params(axis="x", labelbottom=lowest)
        if groups:
            used[0].legend(
                title=args.groupcol,
                fontsize="small",
                title_fontsize="small",
                loc=tuple(args.legendloc) if args.legendloc else "best",
            )
        if args.subgraphcol and args.title:
            fig.suptitle(args.title)
        fig.tight_layout()
        save_or_show(fig, args)
