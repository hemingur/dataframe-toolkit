"""
dftk.commands.wstat_cmd — dftk wstat subcommand.

Port of dfwstat.py: weighted descriptive statistics using
statsmodels.stats.weightstats.DescrStatsW.

Output columns
--------------
  [group(s),] name, weight, totalweight, wsum, wmean, wstd, wmcilo, wmcihi
"""

import argparse

import pandas as pd
import statsmodels.stats.weightstats as ssw

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io


def _wstat(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    weights = list(args.weights)
    if len(weights) == 1:
        weights = weights * len(args.cols)

    if len(args.cols) != len(weights):
        raise ValueError(
            f"Number of weight columns ({len(weights)}) must match "
            f"stat columns ({len(args.cols)}) or be 1."
        )

    alpha = 1.0 - 0.01 * args.confidencelevel
    probs = [alpha / 2, 1.0 - alpha / 2]

    header = [
        "name",
        "weight",
        "totalweight",
        "wsum",
        "wmean",
        "wstd",
        "wmcilo",
        "wmcihi",
    ]
    rows = []

    def _add_cols(sub: pd.DataFrame, prefix: tuple = ()) -> None:
        for c, w in zip(args.cols, weights, strict=False):
            pair = sub[[c, w]].dropna()
            ws = ssw.DescrStatsW(pair[c], pair[w])
            q = ws.quantile(probs)
            rows.append(
                (
                    *prefix,
                    c,
                    w,
                    float(ws.sum_weights),
                    ws.sum,
                    ws.mean,
                    ws.std,
                    float(q.iloc[0]),
                    float(q.iloc[1]),
                )
            )

    if args.groups:
        header = [*args.groups, *header]
        for group, gdf in df.groupby(args.groups):
            prefix = group if isinstance(group, tuple) else (group,)
            _add_cols(gdf, prefix)
    else:
        _add_cols(df)

    return pd.DataFrame(rows, columns=header)


_EPILOG = """\
COLUMN PAIRS (-c / -w)
----------------------
  Each stat column (-c) is paired with a weight column (-w).
  If a single weight column is given it is broadcast to all stat columns:

    dftk wstat data.tsv -c value -w count
    dftk wstat data.tsv -c x y -w wx wy
    dftk wstat data.tsv -c x y -w w          # w used for both x and y

OUTPUT COLUMNS
--------------
  name         stat column name
  weight       weight column name
  totalweight  sum of weights
  wsum         weighted sum
  wmean        weighted mean
  wstd         weighted standard deviation
  wmcilo       lower weighted quantile (confidence level / 2)
  wmcihi       upper weighted quantile (1 - confidence level / 2)

EXAMPLES
--------
  dftk wstat data.tsv -c value -w count
  dftk wstat data.tsv -c value -w count -g group
  dftk wstat data.tsv -c x y -w w --confidencelevel 99
"""


class WstatCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "wstat"

    @property
    def help(self) -> str:
        return "Weighted descriptive statistics (wmean, wstd, weighted quantile CI)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("wstat options")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            required=True,
            metavar="COL",
            help="Column(s) to compute weighted statistics on.",
        )
        g.add_argument(
            "-w",
            "--weights",
            nargs="+",
            required=True,
            metavar="COL",
            help="Weight column(s); one per stat column, or one broadcast to all.",
        )
        g.add_argument(
            "-g",
            "--groups",
            nargs="+",
            default=[],
            metavar="COL",
            help="Group column(s); statistics are computed within each group.",
        )
        g.add_argument(
            "--confidencelevel",
            type=float,
            default=95.0,
            metavar="PCT",
            help="Confidence level in percent for wmcilo/wmcihi (default: 95.0).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        all_cols = list(args.cols) + list(args.weights) + list(args.groups)
        df = io.read(args)
        check_cols(df, all_cols, "-c/-w/-g")

        if df.empty:
            return

        result = _wstat(df, args)
        io.printdf(result, args)
