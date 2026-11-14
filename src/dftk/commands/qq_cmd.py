"""
dftk.commands.qq_cmd — dftk qq subcommand.

QQ plot of p-values: observed -log10(p), sorted, against what n null tests
would give, with a confidence band and the genomic inflation factor lambda.
Takes the p-values the same ways as manhattan (helpers in common/gwas.py).
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
    gwas_columns,
    log_yaxis,
    scatter_capped,
)
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

_POINT_COLOUR = "#1F4E79"  # the manhattan plot's dark blue
_BAND_COLOUR = "#BBBBBB"

_EPILOG = """\
Each point is one test: its -log10(p) (y) against the -log10(p) the same
rank would have if every test were null (x; the i-th smallest of n
uniform p-values averages i/(n+1)).  Points on the diagonal: no signal.  A
few points leaving it at the top right: associations.  The whole curve
lifted off the diagonal from the start: inflation, e.g. population
stratification or relatedness.  The grey band holds 95 % (--ci) of null
points at each rank.

lambda (genomic inflation factor) is the median chi-squared divided by its
null median (0.455 for 1 df), from the p-values whichever column they come
from; it is written on the plot and logged.  About 1: no inflation.  For a
chi-squared column with other degrees of freedom, give --chi2df.

P-values: the Pval column unless named, as in manhattan: --pcol,
--mlogpcol (already -log10(p)) or --chi2col (exact where p underflows to
0).  Points near the origin are thinned to one per cell of a fine grid
(--no-thin to draw all); every point with p < 0.01 is drawn.

EXAMPLES
--------
Basic QQ plot (a Pval column):

  dftk qq data.tsv -f qq.png
  dftk dataset gwas -o | dftk qq -f qq.png

From a chi-squared column, here inflated by 15 % (lambda 1.02 → 1.18):

  dftk dataset gwas -o | dftk eval -f "Chisq = Chisq * 1.15" -o \\
      | dftk qq --chi2col Chisq -f qq.png

Cap the y axis; points above it are drawn as triangles on the top edge:

  dftk dataset gwas -o | dftk qq --ylim 0 20 -f qq.png

Or log-scale it above -log10(p) = 10, so the strongest hits and the bulk
both show:

  dftk dataset gwas -o | dftk qq --chi2col Chisq --logy -f qq.png

Several scans (a study column), or one scan split by allele-frequency bin:
one curve and one lambda each with -g, or a panel each with --subgraphcol:

  dftk qq all_studies.tsv -g study -f qq.png

lambda as a table, without the plot:

  dftk dataset gwas -o | dftk eval -f "chi2 = pval_to_chi2(Pval, 1)" -o \\
      | dftk stat -c chi2 -o | dftk eval -f "lam = median / 0.4549" --select median lam
  (add -g study to stat for one row per study)
"""


def qq_table(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """df with _y = -log10(p); rows without a p-value left out."""
    y, _ = gwas.resolve_neglogp(df, args.pcol, args.mlogpcol, args.chi2col, args.chi2df)
    out = df.assign(_y=y)
    keep = np.isfinite(out["_y"])
    if not keep.all():
        logger.info(
            "Left out %d of %d rows without a p-value", int((~keep).sum()), len(out)
        )
    out = out[keep]
    if out.empty:
        raise ValueError("No rows with a p-value.")
    return out


def _lambda_df(args) -> float:
    """Degrees of freedom for lambda: --chi2df if it is a number, else 1."""
    if args.chi2col is None:
        return 1.0
    try:
        return float(args.chi2df)
    except ValueError:
        return 1.0  # per-row df: lambda on the 1-df scale


def _series(y: np.ndarray, args):
    """(expected, observed) for one curve, largest first, and its lambda."""
    obs = np.sort(y)[::-1]
    exp = gwas.qq_expected(len(obs))
    return exp, obs, gwas.inflation_lambda(obs, _lambda_df(args))


def _draw(ax, data: pd.DataFrame, args, groupcols=None) -> None:
    if groupcols:
        groups = [
            (
                make_grouplabel(name, groupcols, args.groupcolformat),
                gdf["_y"].to_numpy(),
            )
            for name, gdf in subgraph_groups(data, groupcols, args.groupcolorder)
        ]
    else:
        groups = [(args.legend, data["_y"].to_numpy())]

    curves = [(label, *_series(y, args)) for label, y in groups]
    top = max(max(c[1][0], c[2][0]) for c in curves) * 1.05
    lo, hi = 0.0, top
    if args.ylim is not None:
        lo, hi = float_or(args.ylim[0], lo), float_or(args.ylim[1], hi)
    xmax = max(c[1][0] for c in curves) * 1.02

    if args.ci:
        # one band, for the curve with the most tests (the others' are nested)
        n = max(len(c[2]) for c in curves)
        bx, blo, bhi = gwas.qq_band(n, args.ci)
        ax.fill_between(bx, blo, bhi, color=_BAND_COLOUR, alpha=0.5, lw=0, zorder=0)
    # many vertices, so the diagonal bends correctly on a --logy axis
    diag = np.linspace(0, xmax, 200)
    ax.plot(diag, diag, color="black", lw=0.8, zorder=1)

    for n, (label, exp, obs, lam) in enumerate(curves):
        logger.info(
            "lambda %s= %.4f (%d tests)", f"({label}) " if label else "", lam, len(obs)
        )
        keep = gwas.thin_mask(exp, obs) if args.thin else np.ones(len(obs), bool)
        colour = WONG_PALETTE[n % len(WONG_PALETTE)] if groupcols else _POINT_COLOUR
        if groupcols and args.show_lambda:
            label = f"{label} (λ = {lam:.3f})"
        scatter_capped(
            ax, exp[keep], obs[keep], hi, args, color=colour, label=label, zorder=2
        )

    if groupcols or args.legend:
        ax.legend(
            loc=tuple(args.legendloc) if args.legendloc else "lower right",
            title=args.legendtitle,
            markerscale=3,
        )
    if args.show_lambda and not groupcols:
        ax.text(
            0.03, 0.97, f"λ = {curves[0][3]:.3f}", transform=ax.transAxes,
            ha="left", va="top",
        )  # fmt: skip

    ax.set_xlim(0, xmax)
    ax.set_ylim(lo, hi)
    if args.logy is not None:
        log_yaxis(ax, args.logy, lo, hi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel(
        args.xlabel if args.xlabel is not None else r"Expected $-\log_{10}(p)$"
    )
    ax.set_ylabel(
        args.ylabel if args.ylabel is not None else r"Observed $-\log_{10}(p)$"
    )


class QqCommand(BaseCommand):
    name = "qq"
    help = "QQ plot of p-values with a confidence band and genomic inflation (lambda)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_read_arguments(parser)
        add_gwas_column_arguments(parser, position=False)

        g = parser.add_argument_group("qq")
        g.add_argument(
            "--ci",
            type=float,
            default=0.95,
            metavar="LEVEL",
            help="Confidence band level (default: 0.95).",
        )
        g.add_argument(
            "--no-ci", dest="ci", action="store_const", const=None, help="No band."
        )
        g.add_argument(
            "--no-lambda",
            dest="show_lambda",
            action="store_false",
            help="Don't write lambda on the plot (it is still logged).",
        )
        g.add_argument(
            "--logy",
            nargs="?",
            type=float,
            const=10.0,
            metavar="T",
            help="Log scale above observed -log10(p) = T, linear below it "
            "(default T: 10).",
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
            default=6.0,
            metavar="S",
            help="Point area in pt² (default: 6).",
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
        parser.set_defaults(size="5x5")
        add_group_arguments(parser)
        add_legend_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        if args.logy is not None and args.logy <= 0:
            raise ValueError(f"--logy must be above 0, not {args.logy:g}.")
        if args.ci is not None and not 0 < args.ci < 1:
            raise ValueError(f"--ci must be between 0 and 1, not {args.ci}.")
        df = io.read(args, columns=gwas_columns(args, position=False))
        check_cols(df, args.groupcol, "-g/--groupcol")
        check_cols(df, args.subgraphcol, "--subgraphcol")
        data = qq_table(df, args)

        if args.file:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.usetex:
            plt.rc("text", usetex=True)
        apply_style(args)

        if args.subgraphcol is not None:
            groups = subgraph_groups(data, args.subgraphcol, args.subgraphorder)
            nrows, ncols = subgraph_layout(len(groups), args.ncols)
            fig, axes = make_figure(args, nrows=nrows, ncols=ncols)
            for idx, (name, gdf) in enumerate(groups):
                ax = axes[idx // ncols][idx % ncols]
                _draw(ax, gdf, args, args.groupcol)
                ax.set_title(
                    make_grouplabel(name, args.subgraphcol, args.subgraphformat)
                )
            for idx in range(len(groups), nrows * ncols):
                axes[idx // ncols][idx % ncols].set_visible(False)
            if args.title:
                fig.suptitle(args.title)
        else:
            fig, axes = make_figure(args)
            _draw(axes[0][0], data, args, args.groupcol)
            axes[0][0].set_title(args.title or "")
        fig.tight_layout()
        save_or_show(fig, args)
