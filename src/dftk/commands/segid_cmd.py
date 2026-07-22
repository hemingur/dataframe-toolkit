"""
dftk.commands.segid_cmd — dftk segid subcommand.

Port of dfsegid.py: assign a segment ID that increments each time the
value in a column changes, useful for grouping contiguous runs of
identical values (e.g. constant-value stretches in a sorted column).

Rows matching --ignore are excluded from the run comparison and always get
segid 0; they don't break a run spanning across them. Redesigned from the
legacy script's magic sentinel default ('1415926535', digits of pi) to a
plain optional flag — omit --ignore to include every row.
"""

import argparse

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

_EPILOG = """\
EXAMPLE
-------
  col
  A
  A
  B
  B
  A

  dftk segid data.tsv -c col

  col  segid
  A    1
  A    1
  B    2
  B    2
  A    3

With --ignore X, rows where col == X get segid 0 and are skipped when
detecting value changes, so a run continues across them:

  col
  A
  X
  A
  B

  dftk segid data.tsv -c col --ignore X

  col  segid
  A    1
  X    0
  A    1
  B    2
"""


def _segid(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    col = args.col
    dest = args.destcol

    df[dest] = 0

    if args.ignore is not None:
        ignore_value = pd.Series([args.ignore]).astype(df[col].dtype).iloc[0]
        mask = df[col] != ignore_value
    else:
        mask = pd.Series(True, index=df.index)

    subseries = df.loc[mask, col]
    df.loc[mask, dest] = subseries.ne(subseries.shift()).cumsum()
    return df


class SegidCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "segid"

    @property
    def help(self) -> str:
        return "Assign a segment ID that increments on each value change"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("segid options")
        g.add_argument(
            "-c",
            "--col",
            required=True,
            metavar="COL",
            help="Column whose value changes define segment boundaries.",
        )
        g.add_argument(
            "-d",
            "--destcol",
            default="segid",
            metavar="DESTCOL",
            help="Name of the output segment-ID column (default: segid).",
        )
        g.add_argument(
            "--ignore",
            default=None,
            metavar="VALUE",
            help="Column value to exclude from segmenting (gets segid 0, "
            "doesn't break a run spanning across it). Default: none.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, [args.col], "-c/--col")
        result = _segid(df, args)
        io.printdf(result, args)
