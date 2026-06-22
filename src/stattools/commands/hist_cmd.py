"""
stattools.commands.hist_cmd — dfstat hist subcommand.

Histogram or kernel density estimate (KDE).

Port of dfhist.py with improvements:
  - --groupcol  overlaid histograms / KDEs per group (colours from Wong palette)
  - --subgraphcol  splits into a subplot grid
  - --size / --fontsize  publication-quality figure presets
  - --bins / --binwidth  bin control (mutual exclusion preserved)
  - --kde  kernel density estimate via seaborn.kdeplot
  - --stats  add mean ± σ to subplot title
  - -y/--weightcol  renamed from --ycol for clarity (weights column)
"""

import argparse

import numpy as np

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io
from stattools.common.plot import (
    add_figure_arguments,
    add_group_arguments,
    add_legend_arguments,
    apply_labels,
    apply_limits,
    apply_style,
    make_figure,
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

# ---------------------------------------------------------------------------
# Single-axes histogram
# ---------------------------------------------------------------------------


def _hist_ax(ax, df, args, title_suffix: str = ""):
    xcol = args.xcol
    legendloc = tuple(args.legendloc) if args.legendloc else "best"

    # Build bins
    data_all = df[xcol].dropna()
    bins = args.bins
    if args.binwidth is not None and not args.kde:
        lo = (data_all.min() // args.binwidth) * args.binwidth
        hi = data_all.max() + args.binwidth
        bins = np.arange(lo, hi, args.binwidth)

    kde_bw = args.binwidth if args.kde and args.binwidth is not None else "scott"

    hist_kw = dict(
        alpha=0.4,
        cumulative=args.cumulative,
        density=args.normed,
    )

    if args.groupcol is not None:
        for groupname, gdf in subgraph_groups(
            df, args.groupcol, getattr(args, "groupcolorder", None)
        ):
            label = make_grouplabel(
                groupname, args.groupcol, getattr(args, "groupcolformat", None)
            )
            data = gdf[xcol].dropna()
            weights = gdf[args.weightcol].values if args.weightcol else None
            if args.kde:
                import seaborn as sns

                sns.kdeplot(
                    data,
                    bw_method=kde_bw,
                    label=label,
                    cumulative=args.cumulative,
                    cut=0,
                    ax=ax,
                )
            else:
                ax.hist(data, bins=bins, weights=weights, label=label, **hist_kw)
        ax.legend(loc=legendloc, title=args.legendtitle)
    else:
        data = data_all
        weights = df[args.weightcol].values if args.weightcol else None
        label = args.legend or None

        if args.stats:
            mu, sigma = data.mean(), data.std()
            stats_str = f"μ={mu:.3g}, σ={sigma:.3g}"
            existing_title = title_suffix or args.title or ""
            title_suffix = (
                f"{existing_title}  ({stats_str})" if existing_title else stats_str
            )

        if args.kde:
            import seaborn as sns

            sns.kdeplot(
                data,
                bw_method=kde_bw,
                label=label,
                cumulative=args.cumulative,
                cut=0,
                ax=ax,
            )
        else:
            ax.hist(data, bins=bins, weights=weights, label=label, **hist_kw)

        if label:
            ax.legend(loc=legendloc, title=args.legendtitle)

    apply_limits(ax, args)
    apply_labels(
        ax,
        args,
        default_xlabel=xcol or "",
        default_ylabel="density" if args.normed else "count",
    )

    title = args.title or ""
    if title_suffix:
        title = f"{title}  [{title_suffix}]" if title else title_suffix
    ax.set_title(title)

    ax.grid(True, linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
EXAMPLES
--------
Basic histogram (10 bins):

  dfstat hist data.tsv -x value

Fixed bin width:

  dfstat hist data.tsv -x value --binwidth 0.5

Normalised density:

  dfstat hist data.tsv -x value --normed

Cumulative:

  dfstat hist data.tsv -x value --cumulative

KDE (kernel density estimate):

  dfstat hist data.tsv -x value --kde

Grouped (overlaid):

  dfstat hist data.tsv -x value -g group

Subplot grid by condition:

  dfstat hist data.tsv -x value --subgraphcol condition -g group

Add mean ± σ to title:

  dfstat hist data.tsv -x value --stats

Weighted histogram:

  dfstat hist data.tsv -x value -y weight_col

Publication figure (Nature single column, PDF):

  dfstat hist data.tsv -x value --size single --fontsize publication -f fig.pdf
"""


class HistCommand(BaseCommand):
    name = "hist"
    help = "Histogram or kernel density estimate (KDE)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_figure_arguments(parser)
        add_group_arguments(parser)
        add_legend_arguments(parser)

        g = parser.add_argument_group("histogram options")
        g.add_argument(
            "-x", "--xcol", metavar="COL", required=True, help="Column to histogram."
        )
        g.add_argument(
            "-y",
            "--weightcol",
            default=None,
            metavar="COL",
            help="Column to use as observation weights.",
        )
        g.add_argument(
            "-xl",
            "--xlabel",
            default=None,
            metavar="TEXT",
            help="X-axis label (default: column name).",
        )
        g.add_argument(
            "-yl",
            "--ylabel",
            default=None,
            metavar="TEXT",
            help="Y-axis label (default: count or density).",
        )
        g.add_argument(
            "-xm",
            "--xlim",
            nargs=2,
            metavar=("LO", "HI"),
            default=None,
            help="X-axis limits.",
        )
        g.add_argument(
            "-ym",
            "--ylim",
            nargs=2,
            metavar=("LO", "HI"),
            default=None,
            help="Y-axis limits.",
        )
        g.add_argument("-lx", "--logx", action="store_true", help="Logarithmic x axis.")
        g.add_argument("-ly", "--logy", action="store_true", help="Logarithmic y axis.")
        g.add_argument(
            "--xmargin", action="store_true", help="Add 1 %% margin to x limits."
        )
        g.add_argument(
            "--ymargin", action="store_true", help="Add 1 %% margin to y limits."
        )

        g.add_argument(
            "-n",
            "--normed",
            action="store_true",
            help="Normalise so area sums to 1 (density).",
        )
        g.add_argument(
            "-c", "--cumulative", action="store_true", help="Plot cumulative histogram."
        )
        g.add_argument(
            "-k",
            "--kde",
            action="store_true",
            help="Plot kernel density estimate instead of histogram "
            "(uses seaborn.kdeplot).",
        )
        g.add_argument(
            "--stats", action="store_true", help="Add mean ± σ to the subplot title."
        )

        bins = parser.add_argument_group("bin options (mutually exclusive)")
        mx = bins.add_mutually_exclusive_group()
        mx.add_argument(
            "-b",
            "--bins",
            type=int,
            default=10,
            metavar="N",
            help="Number of bins (default: 10).",
        )
        mx.add_argument(
            "-w",
            "--binwidth",
            type=float,
            default=None,
            metavar="W",
            help="Bin width (overrides --bins; also sets KDE bandwidth).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, [args.xcol], "-x/--xcol")
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        if args.weightcol:
            check_cols(df, [args.weightcol], "-y/--weightcol")

        if args.file:
            import matplotlib

            matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        if getattr(args, "usetex", False):
            plt.rc("text", usetex=True)

        apply_style(args)

        if args.subgraphcol is not None:
            groups = subgraph_groups(
                df, args.subgraphcol, getattr(args, "subgraphorder", None)
            )
            nrows, ncols = subgraph_layout(len(groups), args.ncols)
            fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
            for idx, (groupname, gdf) in enumerate(groups):
                ax = axes[idx // ncols][idx % ncols]
                suffix = make_grouplabel(
                    groupname, args.subgraphcol, getattr(args, "subgraphformat", None)
                )
                _hist_ax(ax, gdf, args, title_suffix=suffix)
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
        else:
            fig, axes = make_figure(args)
            _hist_ax(axes[0][0], df, args)
            fig.tight_layout()

        save_or_show(fig, args)
