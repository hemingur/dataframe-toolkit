"""
stattools.commands.print_cmd — the ``dfstat print`` subcommand.

Reads any dfstat-compatible input (TSV file, TSV on stdin, or a parquet pipe
path on stdin) and writes TSV to stdout.  Acts as the terminal stage of a
dfstat pipeline:

    dfstat stat -g grp -s val -o \\
      | dfstat pivot -i grp -v val_mean -o \\
      | dfstat print
"""

import sys
import argparse

from stattools.commands.base import BaseCommand
from stattools.common.io import io


class PrintCommand(BaseCommand):
    """Read any dfstat input and write TSV to stdout."""

    name = "print"
    help = "Convert any input (TSV or parquet pipe) to TSV on stdout"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # Only the read-side arguments — no -o/--output, since the whole
        # point of this command is to force TSV to stdout.
        io.parser_read(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        try:
            df.to_csv(sys.stdout, sep="\t", index=False)
        except (BrokenPipeError, IOError):
            sys.stderr.close()
