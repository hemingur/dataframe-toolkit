"""
dftk.commands.transpose_cmd — dftk transpose subcommand.

Port of dftpos.py: flips rows and columns. Original column names become
values in a new key column (default name: "column"); original row
positions become the new column headers.

Example
-------
Input:

  sample  x  y
  s1      1  2
  s2      3  4

  dftk transpose data.tsv

Output:

  column  0  1
  sample  s1 s2
  x       1  3
  y       2  4

Runnable version — the bundled `longley` dataset, limited to 3 rows via
--sql LIMIT (dataset has no --nrows of its own) so the transposed output
stays legible:

  dftk dataset longley -o | dftk query ... --sql "SELECT * FROM data LIMIT 3" -o \\
      | dftk transpose ...
"""

import argparse

from dftk.commands.base import BaseCommand
from dftk.common.io import io

_EPILOG = __doc__


class TransposeCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "transpose"

    @property
    def help(self) -> str:
        return "Transpose rows and columns"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("transpose options")
        g.add_argument(
            "--keycol",
            default="column",
            metavar="NAME",
            help="Name for the new column holding the original column names "
            "(default: column).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        result = df.transpose().reset_index(names=args.keycol)
        result.columns = [str(c) for c in result.columns]
        io.printdf(result, args)
