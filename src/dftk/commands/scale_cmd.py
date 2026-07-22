"""
dftk.commands.scale_cmd — dftk scale subcommand.

Port of dfscale.py: column-wise normalization / rank transformation /
regression detrending.

Three modes (mutually exclusive flags)
---------------------------------------
Default (scale)   Linear shift + scale.  New options --shift / --scale
                  replace the old --noshift / --minshift / --inishift /
                  --noscale / --sumscale / --maxscale / --meanscale flags.
--rank            Rank normalization.
--resid           Regression detrending (residuals); requires -f/--formula.

Output
------
Each processed column gets a companion column named <col>_scaled appended
to the output dataframe.  All other columns are preserved.

Grouping (-g)
-------------
Any mode can be applied per group.  Results are concatenated into a single
dataframe and written once (cleaner than the old loop-with-removeheader
approach).
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def scalecols(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Linear shift + scale normalization."""
    for col in args.cols:
        newcol = f"{col}_scaled"
        x = df[col].astype(float)

        shift_val = {
            "mean": x.mean(),
            "min": x.min(),
            "first": float(x.iloc[0]),
            "none": 0.0,
        }[args.shift]

        shifted = x - shift_val

        if args.scale_by == "none":
            df[newcol] = shifted
            continue

        scale_val = {
            "std": shifted.std(),
            "sum": shifted.sum(),
            "max": shifted.abs().max(),
            "mean": shifted.mean(),
        }[args.scale_by]

        if args.verbose:
            print(
                f"{newcol}: shift={shift_val:.6g}  scale={scale_val:.6g}",
                file=sys.stderr,
            )

        df[newcol] = shifted / scale_val if scale_val != 0 else shifted

    return df


def rankcols(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Rank normalization to uniform [0,1] or normal scores."""
    n = len(df)
    for col in args.cols:
        newcol = f"{col}_scaled"
        x = df[col].astype(float)
        rnk = x.rank(method="average").values

        if args.rankdist == "uniform":
            df[newcol] = (rnk - 0.5) / n
        else:  # normal (Blom transform)
            p = (rnk - 3 / 8.0) / (n + 1 / 4.0)
            z = norm.ppf(p)
            z = z - z.mean()
            s = z.std(ddof=0)
            df[newcol] = z / s if s > 0 else np.zeros_like(z)

    return df


def residcols(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Regression detrending: replace column with (intercept +) residuals."""
    # Import here to avoid circular imports at module level
    from dftk.commands.fit_cmd import regress_it

    # regress_it needs these attrs; set safe defaults if absent
    if not hasattr(args, "robust"):
        args.robust = False
    if not hasattr(args, "weights"):
        args.weights = None

    res = regress_it(df, args)
    col = res.model.endog_names
    newcol = f"{col}_scaled"

    intercept = 0.0
    if not args.nointercept and "Intercept" in res.params:
        intercept = res.params["Intercept"]

    if args.verbose:
        print(res.summary(), file=sys.stderr)

    df[newcol] = intercept + res.resid
    return df


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
MODES
-----
Default (scale):

  dftk scale data.tsv -c y                        z-score (mean shift, std scale)
  dftk scale data.tsv -c y --shift min            shift so min=0, scale by std
  dftk scale data.tsv -c y --shift min --scale max shift so min=0, scale so max=1
  dftk scale data.tsv -c y --shift none --scale sum no shift, scale so sum=1
  dftk scale data.tsv -c y --scale none           shift to zero mean, no scaling

Rank normalization (--rank):

  dftk scale data.tsv -c y --rank                 normal scores (default)
  dftk scale data.tsv -c y --rank --rankdist uniform  uniform [0, 1]

Regression detrending (--resid):

  dftk scale data.tsv -c y --resid -f "y ~ x"
  dftk scale data.tsv -c y --resid -f "y ~ x" --nointercept

Grouping:

  dftk scale data.tsv -c y -g group               z-score within each group
  dftk scale data.tsv -c y --rank -g group         rank within each group

SHIFT OPTIONS (--shift)
-----------------------
  mean    Subtract the column mean (default).  Results in zero mean.
  min     Subtract the column minimum.  Lowest value becomes 0.
  first   Subtract the first observation.  First value becomes 0.
  none    No shift applied.

SCALE OPTIONS (--scale)
-----------------------
  std     Divide by standard deviation (default).  Results in unit std.
  sum     Divide by sum.  Values sum to 1.
  max     Divide by absolute maximum.  Largest absolute value becomes 1.
  mean    Divide by mean.  Mean becomes 1.
  none    No scaling applied.
"""


class ScaleCommand(BaseCommand):
    name = "scale"
    help = "Column-wise normalization, rank transformation, or regression detrending."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("column selection")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            metavar="COL",
            default=None,
            help="Column(s) to process.  Required for scale and rank modes.",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="Apply transformation independently within each group.",
        )

        mode = parser.add_argument_group("mode (mutually exclusive)")
        mx = mode.add_mutually_exclusive_group()
        mx.add_argument(
            "--rank",
            action="store_true",
            help="Rank normalization (normal scores by default; see --rankdist).",
        )
        mx.add_argument(
            "--resid",
            action="store_true",
            help="Regression detrending: replace column with (intercept +) residuals. "
            "Requires -f/--formula.",
        )

        sp = parser.add_argument_group("scale mode options")
        sp.add_argument(
            "--shift",
            choices=["mean", "min", "first", "none"],
            default="mean",
            help="What to subtract before scaling (default: mean).",
        )
        sp.add_argument(
            "--scale",
            dest="scale_by",
            choices=["std", "sum", "max", "mean", "none"],
            default="std",
            help="What to divide by after shifting (default: std).",
        )

        rp = parser.add_argument_group("rank mode options")
        rp.add_argument(
            "--rankdist",
            choices=["normal", "uniform"],
            default="normal",
            help="Distribution for rank normalization (default: normal = Blom scores).",
        )

        dp = parser.add_argument_group("resid mode options")
        dp.add_argument(
            "-f",
            "--formula",
            default=None,
            metavar="FORMULA",
            help="R-style formula for regression detrending, e.g. 'y ~ x'.",
        )
        dp.add_argument(
            "--nointercept",
            action="store_true",
            help="Return raw residuals without adding back the intercept.",
        )

        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Print shift/scale values (scale mode) or full summary (resid mode) "
            "to stderr.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        # Validate mode / required args
        if args.resid:
            if not args.formula:
                raise ValueError("--resid requires -f/--formula.")
            fn = residcols
        elif args.rank:
            if not args.cols:
                raise ValueError("--rank requires -c/--cols.")
            fn = rankcols
        else:
            if not args.cols:
                raise ValueError("-c/--cols is required.")
            fn = scalecols

        df = io.read(args)
        check_cols(df, args.cols, "-c/--cols")
        check_cols(df, args.groupcol, "-g/--groupcol")

        if args.groupcol is None:
            result = fn(df, args)
            io.printdf(result, args)
        else:
            parts = []
            for groupname, groupdf in df.groupby(args.groupcol):
                if args.verbose:
                    groupstr = (
                        ", ".join(
                            f"{k}={v}"
                            for k, v in zip(args.groupcol, groupname, strict=False)
                            if isinstance(groupname, tuple)
                        )
                        or f"{args.groupcol[0]}={groupname}"
                    )
                    print(f"Group: {groupstr}", file=sys.stderr)
                parts.append(fn(groupdf.copy(), args))
            io.printdf(pd.concat(parts, ignore_index=True), args)
