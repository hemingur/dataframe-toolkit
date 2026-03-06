"""
stattools.commands.melt_cmd — dfstat melt subcommand.

Port of dfmelt.py: wide-to-long reshape via pandas.melt.

Unpivots all columns that are not listed as index columns (-i) into two
new columns: a variable column (column names) and a value column (cell
values).

EXAMPLE
-------
Wide format:

  sample  gene_A  gene_B  gene_C
  s1      1.2     3.4     5.6
  s2      7.8     9.0     1.1

  dfstat melt data.tsv -i sample

Long format output:

  sample  variable  value
  s1      gene_A    1.2
  s1      gene_B    3.4
  s1      gene_C    5.6
  s2      gene_A    7.8
  …
"""

import argparse

from stattools.commands.base import BaseCommand
from stattools.common.io import io, check_cols


class MeltCommand(BaseCommand):
    name = "melt"
    help = "Reshape wide-to-long (unpivot) via pandas.melt."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.add_io_arguments(parser)

        g = parser.add_argument_group("melt options")
        g.add_argument(
            "-i", "--indexcols",
            nargs="+",
            default=None,
            metavar="COL",
            help="Column(s) to keep as row identifiers (id_vars). "
                 "All other columns are melted.",
        )
        g.add_argument(
            "-d", "--destcol",
            default="variable",
            metavar="NAME",
            help="Name of the new column that holds the original column names "
                 "(default: variable).",
        )
        g.add_argument(
            "-v", "--valuecol",
            default="value",
            metavar="NAME",
            help="Name of the new column that holds the cell values "
                 "(default: value).",
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
