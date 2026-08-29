"""
dftk.commands.chi_cmd — dftk chi subcommand.

Corner-resolved chi-plot: a visual diagnostic for which direction (if any)
two variables have tail dependence in, complementing the numeric
lambda_lower/lambda_upper columns from `dftk corr --tail`.

Follows the structure of scat_cmd.py -- same --groupcol / --subgraphcol /
--size / --fontsize figure conventions as the rest of the Plots group.
"""

import argparse

import numpy as np

from dftk.commands.base import BaseCommand
from dftk.common.copula import corner_chi, pseudo_observations
from dftk.common.io import check_cols, io
from dftk.common.plot import (
    add_figure_arguments,
    add_group_arguments,
    add_legend_arguments,
    add_xy_arguments,
    apply_labels,
    apply_style,
    make_figure,
    make_grouplabel,
    save_or_show,
    subgraph_groups,
    subgraph_layout,
)

# ---------------------------------------------------------------------------
# Single-axes chi-plot (reused per subplot)
# ---------------------------------------------------------------------------


def _chi_ax(ax, df, args, title_suffix: str = ""):
    xcol, ycol = args.xcol, args.ycol
    legendloc = tuple(args.legendloc) if getattr(args, "legendloc", None) else "best"

    def _draw(sub_df, label=None, **scatter_kw):
        a, b = sub_df[[xcol, ycol]].dropna().values.T
        if len(a) < 3:
            return
        u, v = pseudo_observations(a, b)
        x, chi = corner_chi(u, v)
        mask = np.abs(x) < 0.8  # trim the noisiest extreme points, standard practice
        ax.scatter(x[mask], chi[mask], s=8, alpha=0.4, label=label, **scatter_kw)
        ci = 1.78 / np.sqrt(len(u))
        return ci

    if args.groupcol is not None:
        # Each group gets its OWN chi statistic computed within its own subset
        # (mixing groups into one leave-one-out CDF would misrepresent both) --
        # overlaid here the same way scat/hist overlay per-group series.
        cis = []
        for groupname, gdf in subgraph_groups(
            df, args.groupcol, getattr(args, "groupcolorder", None)
        ):
            label = make_grouplabel(
                groupname, args.groupcol, getattr(args, "groupcolformat", None)
            )
            ci = _draw(gdf, label=label)
            if ci is not None:
                cis.append(ci)
        ax.legend(loc=legendloc, title=getattr(args, "legendtitle", None))
        ci = max(cis) if cis else 0.0
    else:
        ci = _draw(df) or 0.0

    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    if ci:
        ax.axhline(ci, color="crimson", lw=1, ls="--", alpha=0.7)
        ax.axhline(-ci, color="crimson", lw=1, ls="--", alpha=0.7)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.2, 1)

    apply_labels(
        ax,
        args,
        default_xlabel="<- lower corner    |    upper corner ->",
        default_ylabel=r"$\chi$ (local dependence)",
    )
    if title_suffix:
        existing = ax.get_title()
        ax.set_title(f"{existing}  [{title_suffix}]" if existing else title_suffix)

    ax.grid(True, linewidth=0.4, alpha=0.5)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
Reads as: chi elevated (and rising) on the LEFT (negative x) only => lower-
tail dependence -- the two variables' worst values tend to co-occur, but
their best values don't (e.g. joint crashes but not joint rallies). Elevated
on the RIGHT only => upper-tail dependence (joint highs, not joint lows).
Elevated similarly on BOTH sides, decaying back toward 0 well before the
edges => generic correlation with no real tail dependence. Flat at 0
everywhere => independence. Dashed red lines mark an approximate 95%
control limit for "indistinguishable from independence".

This is a visual complement to `dftk corr --tail`'s numeric
lambda_lower/lambda_upper columns -- same underlying question, this command
answers it by eye and per-corner in one plot instead of at a chosen set of
thresholds. Needs a reasonable amount of data to be legible: a few hundred
points is usually enough to see the shape, but a strong one-sided asymmetry
is easiest to trust with several thousand.

EXAMPLES
--------
Basic chi-plot:

  dftk chi data.tsv -x recomb_rate -y gc_content

Split into a subplot grid by category, to check whether tail dependence is
consistent across the whole dataset or driven by a few subgroups (e.g.
chromosome, sector, region):

  dftk chi data.tsv -x equity_a_return -y equity_b_return --subgraphcol sector

Colour-code by group instead of splitting into subplots:

  dftk chi data.tsv -x rainfall -y runoff -g basin

Publication figure (Nature single column, PDF):

  dftk chi data.tsv -x x -y y --size single --fontsize publication -f fig.pdf
"""


class ChiCommand(BaseCommand):
    name = "chi"
    help = "Corner-resolved chi-plot: visualize lower vs upper tail dependence."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_xy_arguments(parser)
        add_figure_arguments(parser)
        add_group_arguments(parser)
        add_legend_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        if not args.xcol or not args.ycol:
            raise ValueError("--xcol (-x) and --ycol (-y) are required.")

        df = io.read(args)
        check_cols(df, [args.xcol, args.ycol], "-x/-y")
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")

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
                _chi_ax(ax, gdf, args, title_suffix=suffix)
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            fig.tight_layout()
        else:
            fig, axes = make_figure(args)
            _chi_ax(axes[0][0], df, args)
            fig.tight_layout()

        save_or_show(fig, args)
