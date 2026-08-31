"""
dftk.commands.melt_cmd — dftk melt subcommand.

Port of dfmelt.py: wide-to-long reshape via pandas.melt.

Unpivots all columns that are not listed as index columns (-i) into two
new columns: a variable column (column names) and a value column (cell
values).

EXAMPLE
-------
Wide format (bundled `elnino` dataset: one row per year, one column per
month):

  YEAR    JAN    FEB    MAR   ...
  1950    23.11  24.20  25.37 ...
  1951    24.19  25.29  25.62 ...

  dftk melt data.tsv -i YEAR

Long format output:

  YEAR    variable  value
  1950    JAN       23.11
  1950    FEB       24.20
  1950    MAR       25.37
  1951    JAN       24.19
  …

Runnable form:

  dftk dataset elnino -o | dftk melt ... -i YEAR
"""

import argparse

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

_EPILOG = __doc__


class MeltCommand(BaseCommand):
    name = "melt"
    help = "Reshape wide-to-long (unpivot) via pandas.melt."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("melt options")
        g.add_argument(
            "-i",
            "--indexcols",
            nargs="+",
            default=None,
            metavar="COL",
            help="Column(s) to keep as row identifiers (id_vars). "
            "All other columns are melted.",
        )
        g.add_argument(
            "-d",
            "--destcol",
            default="variable",
            metavar="NAME",
            help="Name of the new column that holds the original column names "
            "(default: variable).",
        )
        g.add_argument(
            "-v",
            "--valuecol",
            default="value",
            metavar="NAME",
            help="Name of the new column that holds the cell values (default: value).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.indexcols, "-i/--indexcols")

        result = df.melt(
            id_vars=args.indexcols,
            var_name=args.destcol,
            value_name=args.valuecol,
        )
        io.printdf(result, args)
