"""
dftk.commands.forest_cmd — dftk forest subcommand.

Forest plot: one row per estimate, drawn as a point with a horizontal
confidence interval, labels down the left, a reference line at no effect
and, with --text, the numbers in columns on the right.  For regression
coefficients (fit output), PheWAS or burden results, subgroup and
meta-analysis tables.
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
    apply_style,
    make_figure,
    parse_figsize,
    save_or_show,
)

logger = logging.getLogger(__name__)

_POINT_COLOUR = "black"
_REF_COLOUR = "#888888"
_ROW_HEIGHT = 0.28  # inches per row when --size isn't given
_TEXT_GAP = 0.25  # inches between text columns (and the arrows)

_EPILOG = """\
Input: one row per estimate, with a label, the estimate and either the
two ends of its confidence interval (--ci LO,HI) or its standard error
(--se SE, interval estimate ± z·SE for --level).  Rows are drawn top to
bottom in file order: sort first (--sortasc in the command before, or
query) to order them.  Rows with no estimate keep their label and draw
nothing.

--exp draws exp() of estimate and interval on a log axis with the
reference line at 1: log odds ratios, log hazard ratios and coefficients
of a log-scale model become ratios.  For a table that already holds
ratios, use --logx instead.  --ref moves the reference line (default 0,
or 1 with --exp or --logx); --ref none removes it.

-g draws several series on each row, side by side and in colour (e.g.
heterozygous vs homozygous carriers, two cohorts, two masks); a label
must then appear once per series.  --section puts rows under bold
headers (e.g. organ systems), in order of first appearance.

--text writes the estimate and interval to the right of the plot, and
--textcol adds columns as they are (p-value, number of carriers, ...).

EXAMPLES
--------
Basic forest plot of a results table (phenotype, beta, se):

  dftk forest results.tsv -y phenotype -x beta --se se -f forest.png

Regression coefficients from fit, with their 95 % intervals:

  dftk dataset tips -o \\
      | dftk fit -f "tip ~ total_bill + size + C(day) + C(smoker)" \\
          --postquery "marker != 'Intercept'" -o \\
      | dftk forest -y marker -x effect --ci cilo,cihi -f coef.png

A log-scale model gives ratios: exp() of the coefficients, on a log
axis, with the numbers and p-values on the right:

  dftk dataset diamonds -o \\
      | dftk fit -f "np.log(price) ~ np.log(carat) + C(cut) + C(color)" \\
          --postquery "marker.str.startswith('C(')" -o \\
      | dftk forest -y marker -x effect --se stdev --exp --text --textcol pval \\
          -xl "price ratio" -f ratios.png

The top hits of a phenome-wide scan of loss-of-function carriers
(dataset phewas is simulated), as odds ratios with their p-values:

  dftk dataset phewas -o \\
      | dftk query -q "genotype == 'het' and Pval < 0.01" --sortasc Pval -o \\
      | dftk forest -y phenotype -x beta --se se --exp --text --textcol Pval \\
          -xl "odds ratio" -f top.png

Heterozygous and homozygous carriers side by side, under category
headers; the few homozygotes give wide intervals, cut off at 0.1 and 20:

  dftk dataset phewas -o \\
      | dftk query -q "category in ['Cardiovascular', 'Haematology']" -o \\
      | dftk forest -y phenotype -x beta --se se --exp -g genotype \\
          --section category --text -xm 0.1 20 -xl "odds ratio" -f carriers.png

Under section headers, one series per sex (penguin body mass by species
and island, mean ± 1.96 SEM):

  dftk dataset penguins -o \\
      | dftk stat -c body_mass_g -g species island sex -o \\
      | dftk forest -y island -x mean --se sem --section species -g sex \\
          --ref none --text --digits 0 -f penguins.png
"""


def _ci_columns(args) -> list[str]:
    """The two column names of --ci LO,HI ([] without it)."""
    if not args.ci:
        return []
    cols = [c.strip() for c in args.ci.split(",")]
    if len(cols) != 2 or not all(cols):
        raise ValueError(
            f"--ci takes two column names separated by a comma, e.g. "
            f"--ci cilo,cihi; not {args.ci!r}."
        )
    return cols


def _z(level: float) -> float:
    from scipy.stats import norm

    return float(norm.ppf(0.5 + level / 2))


def _numeric(df: pd.DataFrame, col: str, option: str) -> np.ndarray:
    values = pd.to_numeric(df[col], errors="coerce")
    bad = values.isna() & df[col].notna()
    if bad.any():
        example = df.loc[bad, col].iloc[0]
        raise ValueError(f"{option} column {col!r} isn't numeric (e.g. {example!r}).")
    return values.to_numpy(dtype=float)


def estimates(df: pd.DataFrame, args) -> pd.DataFrame:
    """_est, _lo, _hi on the plotted scale, and _se (NaN without --se/--ci)."""
    est = _numeric(df, args.xcol, "-x/--xcol")
    if args.se:
        se = _numeric(df, args.se, "--se")
        if (se < 0).any():
            raise ValueError(f"--se column {args.se!r} has negative values.")
        z = _z(args.level)
        lo, hi = est - z * se, est + z * se
    elif args.ci:
        lo_col, hi_col = _ci_columns(args)
        lo = _numeric(df, lo_col, "--ci")
        hi = _numeric(df, hi_col, "--ci")
        # for --weighted: half the interval's width, in place of the SE
        se = np.abs(hi - lo) / 2
    else:
        lo = hi = se = np.full(len(df), np.nan)
    if args.exp:
        est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
    return pd.DataFrame({"_est": est, "_lo": lo, "_hi": hi, "_se": se}, index=df.index)


def layout(df: pd.DataFrame, args) -> tuple[pd.Series, list[tuple[float, str, bool]]]:
    """Y position of every row of *df*, and the (y, text, is_header) of
    each label on the axis.  Rows run 0, 1, ... from the top."""
    label = df[args.ycol].astype(str).where(df[args.ycol].notna(), "")
    section = (
        df[args.section].astype(str).where(df[args.section].notna(), "")
        if args.section
        else pd.Series("", index=df.index)
    )
    sections = list(dict.fromkeys(section))
    groups = _group_order(df, args)

    ypos = pd.Series(np.nan, index=df.index)
    ticks: list[tuple[float, str, bool]] = []
    slot = 0
    for sec in sections:
        rows = df.index[section == sec]
        if args.section:
            ticks.append((slot, sec, True))
            slot += 1
        if not args.groupcol:
            for row in rows:
                ypos[row] = slot
                ticks.append((slot, label[row], False))
                slot += 1
            continue
        group = df.loc[rows, args.groupcol].astype(str)
        dup = pd.DataFrame({"l": label[rows], "g": group}).duplicated()
        if dup.any():
            row = rows[dup.to_numpy()][0]
            where = f" in section {sec!r}" if args.section else ""
            raise ValueError(
                f"Label {label[row]!r}{where} appears more than once for "
                f"{args.groupcol} = {group[row]!r}; with -g each label needs "
                f"one row per series."
            )
        offsets = _offsets(len(groups))
        for lab in dict.fromkeys(label[rows]):
            for row in rows[(label[rows] == lab).to_numpy()]:
                ypos[row] = slot + offsets[groups.index(group[row])]
            ticks.append((slot, lab, False))
            slot += 1
    return ypos, ticks


def _group_order(df: pd.DataFrame, args) -> list[str]:
    if not args.groupcol:
        return []
    found = list(dict.fromkeys(df[args.groupcol].astype(str)))
    if not args.groupcolorder:
        return found
    listed = [g for g in args.groupcolorder if g in found]
    return listed + [g for g in found if g not in listed]


def _offsets(n: int) -> list[float]:
    """Y offsets of n series within a row, the first at the top."""
    if n <= 1:
        return [0.0]
    spread = {2: 0.3, 3: 0.5}.get(n, 0.6)  # share of the row they take
    step = spread / (n - 1)
    return [-spread / 2 + i * step for i in range(n)]


def _fmt(x: float, digits: int) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


def ci_text(est: float, lo: float, hi: float, digits: int) -> str:
    """'1.32 [1.05, 1.66]'; just the estimate without an interval."""
    if not np.isfinite(est):
        return ""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return _fmt(est, digits)
    return f"{_fmt(est, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def cell_text(value) -> str:
    """A --textcol value: integers as they are, other numbers to 3
    significant digits, text unchanged, missing as empty."""
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.3g}"
    return str(value)


def _est_header(args) -> str:
    name = f"exp({args.xcol})" if args.exp else args.xcol
    if args.se:
        return f"{name} [{args.level * 100:g}% CI]"
    if args.ci:
        lo, hi = _ci_columns(args)
        if args.exp:
            lo, hi = f"exp({lo})", f"exp({hi})"
        return f"{name} [{lo}, {hi}]"
    return name


def _sizes(est: pd.DataFrame, args) -> np.ndarray:
    base = args.markersize
    if not args.weighted:
        return np.full(len(est), base)
    se = est["_se"].to_numpy(dtype=float)
    with np.errstate(divide="ignore"):
        w = 1.0 / se**2
    w[~np.isfinite(w)] = np.nan
    if np.isnan(w).all():
        return np.full(len(est), base)
    rel = np.clip(w / np.nanmax(w), 0.1, 1.0)
    rel[np.isnan(rel)] = 0.1
    return base * 3 * rel


def _draw(ax, df: pd.DataFrame, est: pd.DataFrame, ypos: pd.Series, args) -> None:
    groups = _group_order(df, args)
    colours = _colours(df, args)
    sizes = pd.Series(_sizes(est, args), index=df.index)
    xlim = _xlim(args)

    series = (
        [(g, df.index[(df[args.groupcol].astype(str) == g).to_numpy()]) for g in groups]
        if groups
        else [(None, df.index)]
    )
    for g, rows in series:
        colour = colours.get(g, _POINT_COLOUR)
        e = est.loc[rows]
        y = ypos[rows].to_numpy()
        lo, hi = e["_lo"].to_numpy(), e["_hi"].to_numpy()
        has_ci = np.isfinite(lo) & np.isfinite(hi)
        lo_c, hi_c = lo.copy(), hi.copy()
        if xlim is not None:
            lo_c = np.clip(lo_c, *xlim)
            hi_c = np.clip(hi_c, *xlim)
        ax.hlines(
            y[has_ci],
            lo_c[has_ci],
            hi_c[has_ci],
            color=colour,
            lw=args.linewidth,
            zorder=2,
        )
        if xlim is not None:  # arrows where an interval runs off the axis
            left = has_ci & (lo < xlim[0])
            right = has_ci & (hi > xlim[1])
            ax.scatter(
                np.full(left.sum(), xlim[0]),
                y[left],
                marker="<",
                s=20,
                color=colour,
                zorder=3,
                clip_on=False,
            )
            ax.scatter(
                np.full(right.sum(), xlim[1]),
                y[right],
                marker=">",
                s=20,
                color=colour,
                zorder=3,
                clip_on=False,
            )
        x = e["_est"].to_numpy()
        show = np.isfinite(x)
        if xlim is not None:
            show &= (x >= xlim[0]) & (x <= xlim[1])
        ax.scatter(
            x[show],
            y[show],
            s=sizes[rows].to_numpy()[show],
            marker=args.marker,
            color=colour,
            edgecolors="none",
            zorder=4,
            label=g if g is not None else None,
        )


def _xlim(args) -> tuple[float, float] | None:
    if args.xlim is None:
        return None
    lo, hi = (float(v) for v in args.xlim)
    if lo >= hi:
        raise ValueError(f"--xlim LO must be below HI, not {lo:g} {hi:g}.")
    if (args.exp or args.logx) and lo <= 0:
        raise ValueError("--xlim must be above 0 on a log axis (--exp, --logx).")
    return lo, hi


def _ref(args) -> float | None:
    if args.ref is None:
        return 1.0 if (args.exp or args.logx) else 0.0
    if args.ref.lower() == "none":
        return None
    try:
        return float(args.ref)
    except ValueError:
        raise ValueError(f"--ref takes a number or none, not {args.ref!r}.") from None


def _text_columns(df, est, ypos, args) -> list:
    """(header, [(y, text, colour), ...]) for each text column; with -g
    each line is in its series' colour."""
    colours = _colours(df, args)
    colour = [
        colours.get(g, _POINT_COLOUR)
        for g in (df[args.groupcol].astype(str) if args.groupcol else [None] * len(df))
    ]
    cols = []
    if args.text:
        cells = [
            (
                ypos[row],
                ci_text(*est.loc[row, ["_est", "_lo", "_hi"]], args.digits),
                colour[i],
            )
            for i, row in enumerate(df.index)
        ]
        cols.append((_est_header(args), cells))
    for col in args.textcol:
        cells = [
            (ypos[row], cell_text(df.at[row, col]), colour[i])
            for i, row in enumerate(df.index)
        ]
        cols.append((col, cells))
    return cols


def _colours(df: pd.DataFrame, args) -> dict[str, str]:
    """Colour of each -g series ({} without -g)."""
    return {
        g: WONG_PALETTE[i % len(WONG_PALETTE)]
        for i, g in enumerate(_group_order(df, args))
    }


def _draw_text(fig, ax, columns, args) -> None:
    """Write the text columns to the right of *ax*, each as wide as its
    widest cell, and make room for them in the figure."""
    if not columns:
        return
    from matplotlib.transforms import ScaledTranslation, blended_transform_factory

    renderer = fig.canvas.get_renderer()
    base = blended_transform_factory(ax.transAxes, ax.transData)
    head = ax.transAxes
    small = "small" if args.groupcol else None
    offset = _TEXT_GAP
    for header, cells in columns:
        shift = ScaledTranslation(offset, 0, fig.dpi_scale_trans)
        items = [
            ax.text(
                1.0,
                y,
                s,
                transform=base + shift,
                va="center",
                ha="left",
                fontsize=small,
                color=colour,
                clip_on=False,
            )
            for y, s, colour in cells
            if s
        ]
        items.append(
            ax.text(
                1.0,
                1.0,
                header,
                transform=head + shift,
                va="bottom",
                ha="left",
                fontweight="bold",
                clip_on=False,
            )
        )
        width = max(t.get_window_extent(renderer).width for t in items) / fig.dpi
        offset += width + _TEXT_GAP
    # the figure grows to the right by the text's width; the axes keep theirs
    w, h = fig.get_size_inches()
    fig.set_size_inches(w + offset, h)
    fig.tight_layout(rect=(0, 0, w / (w + offset), 1))


class ForestCommand(BaseCommand):
    name = "forest"
    help = "Forest plot: estimates with confidence intervals, one row each."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        c = parser.add_argument_group("columns")
        c.add_argument(
            "-x", "--xcol", required=True, metavar="COL", help="Estimate column."
        )
        c.add_argument(
            "-y",
            "--ycol",
            required=True,
            metavar="COL",
            help="Label column: the text down the left, one row each.",
        )
        ci = c.add_mutually_exclusive_group()
        ci.add_argument(
            "--ci",
            metavar="LO,HI",
            help="Interval columns, e.g. --ci cilo,cihi (on the estimate's scale).",
        )
        ci.add_argument(
            "--se",
            metavar="COL",
            help="Standard-error column: interval estimate ± z·SE (see --level).",
        )
        c.add_argument(
            "--level",
            type=float,
            default=0.95,
            metavar="L",
            help="Confidence level of the interval with --se (default: 0.95).",
        )
        c.add_argument(
            "-g",
            "--groupcol",
            metavar="COL",
            help="Several series per row, side by side, one colour each.",
        )
        c.add_argument(
            "--groupcolorder",
            nargs="+",
            default=None,
            metavar="VAL",
            help="Series order for -g (top to bottom within a row).",
        )
        c.add_argument(
            "--section",
            metavar="COL",
            help="Group rows under bold headers, in order of first appearance.",
        )

        f = parser.add_argument_group("forest")
        f.add_argument(
            "--exp",
            action="store_true",
            help="Draw exp() of the estimate and interval on a log axis, "
            "reference at 1 (log odds ratios to odds ratios).",
        )
        f.add_argument(
            "--ref",
            metavar="V|none",
            help="Reference line (default: 0, or 1 with --exp or --logx).",
        )
        f.add_argument(
            "--text",
            action="store_true",
            help="Write 'estimate [lo, hi]' in a column right of the plot.",
        )
        f.add_argument(
            "--textcol",
            nargs="+",
            default=[],
            metavar="COL",
            help="More columns to write on the right, as they are "
            "(e.g. Pval n_carriers).",
        )
        f.add_argument(
            "--digits",
            type=int,
            default=2,
            metavar="N",
            help="Decimals in the --text column (default: 2).",
        )
        f.add_argument(
            "--weighted",
            action="store_true",
            help="Marker area by precision, 1/SE² (from --se, or the "
            "interval's width with --ci).",
        )
        f.add_argument(
            "-m",
            "--marker",
            default="s",
            metavar="M",
            help="Matplotlib marker (default: s, a square).",
        )
        f.add_argument(
            "--markersize",
            type=float,
            default=25.0,
            metavar="S",
            help="Marker area in pt² (default: 25).",
        )
        f.add_argument(
            "--linewidth",
            type=float,
            default=1.2,
            metavar="W",
            help="Interval line width (default: 1.2).",
        )

        a = parser.add_argument_group("axes")
        a.add_argument(
            "-xl", "--xlabel", metavar="TEXT", help="X-axis label (default: -x)."
        )
        a.add_argument(
            "-xm",
            "--xlim",
            nargs=2,
            metavar=("LO", "HI"),
            help="X-axis limits; intervals beyond them end in an arrow.",
        )
        a.add_argument(
            "-lx",
            "--logx",
            action="store_true",
            help="Log x axis, for ratios already in the table (reference at 1).",
        )
        a.add_argument(
            "--legendloc",
            nargs=2,
            type=float,
            metavar=("X", "Y"),
            help="Position of the -g legend, in axes coordinates (default: best).",
        )

        add_figure_arguments(parser)
        parser.set_defaults(size=None)

    def execute(self, args: argparse.Namespace) -> None:
        if not 0 < args.level < 1:
            raise ValueError(f"--level must be between 0 and 1, not {args.level:g}.")
        if args.digits < 0:
            raise ValueError(f"--digits must be 0 or more, not {args.digits}.")
        if args.exp and args.logx:
            raise ValueError("--exp already draws a log axis; leave out --logx.")
        ci_cols = _ci_columns(args)
        df = io.read(args, columns=option_columns(args, *ci_cols))
        check_cols(df, [args.xcol], "-x/--xcol")
        check_cols(df, [args.ycol], "-y/--ycol")
        if args.se:
            check_cols(df, [args.se], "--se")
        if ci_cols:
            check_cols(df, ci_cols, "--ci")
        if args.groupcol:
            check_cols(df, [args.groupcol], "-g/--groupcol")
        if args.section:
            check_cols(df, [args.section], "--section")
        check_cols(df, args.textcol, "--textcol")
        if df.empty:
            raise ValueError("No rows to plot.")
        df = df.reset_index(drop=True)
        if args.weighted and not (args.se or args.ci):
            raise ValueError("--weighted needs --se or --ci.")

        est = estimates(df, args)
        if (args.logx or args.exp) and ((est[["_est", "_lo", "_hi"]] <= 0).any().any()):
            raise ValueError(
                "--logx needs values above 0; for log ratios use --exp instead."
            )
        ypos, ticks = layout(df, args)
        xlim = _xlim(args)
        ref = _ref(args)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)
        n_slots = len(ticks)
        if args.size is None:
            per_row = _ROW_HEIGHT * max(1.0, 0.6 * len(_group_order(df, args)))
            args.size = f"7x{0.9 + per_row * n_slots:.2f}"
            fig, ax = plt.subplots(figsize=parse_figsize(args.size))
        else:
            fig, axes = make_figure(args)
            ax = axes[0][0]

        _draw(ax, df, est, ypos, args)
        if ref is not None:
            ax.axvline(ref, color=_REF_COLOUR, lw=0.8, ls="--", zorder=1)
        if args.exp or args.logx:
            ax.set_xscale("log")
            _log_ticks(ax, est, xlim, ref)
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_ylim(n_slots - 0.5, -0.5)  # first row at the top
        ax.set_yticks([t[0] for t in ticks])
        ax.set_yticklabels([t[1] for t in ticks])
        for label, (_, _, header) in zip(ax.get_yticklabels(), ticks, strict=True):
            if header:
                label.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_xlabel(
            args.xlabel
            if args.xlabel is not None
            else (f"exp({args.xcol})" if args.exp else args.xcol)
        )
        if args.groupcol:
            ax.legend(
                title=args.groupcol,
                fontsize="small",
                title_fontsize="small",
                loc=tuple(args.legendloc) if args.legendloc else "best",
            )
        ax.set_title(args.title or "")
        fig.tight_layout()
        _draw_text(fig, ax, _text_columns(df, est, ypos, args), args)
        save_or_show(fig, args)


def _log_ticks(ax, est: pd.DataFrame, xlim, ref) -> None:
    """Plain-number ticks on the log axis: 0.5, 1, 2 rather than 10^-0.3.

    Within a decade, evenly spaced ticks (the log axis would label its
    minor ticks on top of each other); over more, 1-2-5 steps."""
    from matplotlib.ticker import LogLocator, MaxNLocator, NullFormatter

    if xlim is not None:
        lo, hi = xlim
    else:
        values = est[["_est", "_lo", "_hi"]].to_numpy(dtype=float).ravel()
        values = values[np.isfinite(values)]
        if ref is not None:
            values = np.append(values, ref)
        lo, hi = values.min(), values.max()
    if hi / lo < 10:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(lambda x, _pos: f"{x:g}")
    ax.xaxis.set_minor_formatter(NullFormatter())
