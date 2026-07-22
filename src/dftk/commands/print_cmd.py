"""
dftk.commands.print_cmd — the ``dftk print`` subcommand.

Reads any dftk-compatible input (TSV file, TSV on stdin, or a parquet pipe
path on stdin) and writes TSV to stdout.  Acts as the terminal stage of a
dftk pipeline:

    dftk stat -g grp -s val -o \\
      | dftk pivot -i grp -v val_mean -o \\
      | dftk print
"""

import argparse
import sys

from dftk.commands.base import BaseCommand
from dftk.common.io import io


class PrintCommand(BaseCommand):
    """Read any dftk input and write TSV to stdout."""

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
        except (OSError, BrokenPipeError):
            sys.stderr.close()
