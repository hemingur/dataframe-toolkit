"""
stattools.commands.binx_cmd — dfstat binx subcommand.

Port of dfbinx.py: assign each row a bin index based on a column value and
a set of bin edges, with an optional replacement by the lower, middle, or
upper edge value of that bin.
"""

import argparse

import numpy as np
import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io


def _parse_binspec(binspec: str) -> list[float]:
    """Parse 'min:max:step' or 'e1,e2,...,eN' into a list of bin edges."""
    if "," in binspec:
        return [float(x) for x in binspec.split(",")]
    parts = binspec.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"binspec must be 'min:max:step' or comma-separated edges, got: {binspec!r}"
        )
    bmin, bmax, bstep = float(parts[0]), float(parts[1]), float(parts[2])
    return list(np.arange(bmin, bmax, bstep))


def _binx(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    bins = _parse_binspec(args.binspec)
    if len(bins) < 2:
        raise ValueError(
            f"binspec must produce at least 2 edges; got {len(bins)}: {bins}"
        )

    dest = args.destcol or f"{args.col}_bin"
    df[dest] = pd.cut(df[args.col], bins, labels=False, include_lowest=True)

    if args.usevalue is not None:
        frac = {"l": 0.0, "m": 0.5, "u": 1.0}[args.usevalue]
        edges = bins

        def _to_edge(idx: float) -> float:
            if pd.isna(idx):
                return float("nan")
            i = int(idx)
            return (1.0 - frac) * edges[i] + frac * edges[i + 1]

        df[dest] = df[dest].apply(_to_edge)

    return df


_EPILOG = """\
BIN SPECIFICATION (--binspec)
------------------------------
  min:max:step   Generate edges with np.arange(min, max, step).
                 Note: max is exclusive, same as Python range().
  e1,e2,...,eN   Explicit comma-separated edge values.

  Examples:
    0:100:10      edges [0, 10, 20, ..., 90]  → 9 bins
    0,5,10,50     3 bins: [0,5), [5,10), [10,50]

USEVALUE (--usevalue)
---------------------
  Replaces the integer bin index with the actual edge value:
    l   lower edge of the bin
    m   midpoint between lower and upper edges
    u   upper edge of the bin

EXAMPLES
--------
  dfstat binx data.tsv -c age -b 0:100:10
  dfstat binx data.tsv -c score -b 0,25,50,75,100 -d quartile --usevalue m
  dfstat binx data.tsv -c x -b -3:4:1 -d x_bin -o | dfstat stat -g x_bin -s y
"""


class BinxCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "binx"

    @property
    def help(self) -> str:
        return "Assign bin indices to a column based on explicit or generated edges."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("binx options")
        g.add_argument(
            "-c",
            "--col",
            required=True,
            metavar="COL",
            help="Column to bin.",
        )
        g.add_argument(
            "-d",
            "--destcol",
            default=None,
            metavar="DESTCOL",
            help="Name of the output bin column (default: <col>_bin).",
        )
        g.add_argument(
            "-b",
            "--binspec",
            required=True,
            metavar="SPEC",
            help="Bin edges: 'min:max:step' or 'e1,e2,...,eN'.",
        )
        g.add_argument(
            "--usevalue",
            choices=["l", "m", "u"],
            default=None,
            help="Replace bin index with bin edge value: l=lower, m=mid, u=upper.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, [args.col], "-c/--col")
        result = _binx(df, args)
        io.printdf(result, args)
