"""
dftk.commands.print_cmd — the ``dftk print`` subcommand.

Reads any dftk-compatible input (TSV file, TSV on stdin, or a parquet pipe
path on stdin) and writes TSV to stdout, or JSON, a Markdown table or a
LaTeX tabular with --format json|jsonl|md|tex.  Acts as the terminal stage of a
dftk pipeline:

    dftk stat -g grp -s val -o \\
      | dftk pivot -i grp -v val_mean -o \\
      | dftk print

With --noheader and --move it names the columns of a headerless file:

    dftk print data.txt --noheader --move V1:chr V2:begin V3:end

A table for a document:

    dftk dataset tips -o | dftk pivot -i day -v tip -f mean -o \\
      | dftk print --digits 3 --format md
"""

import argparse

from dftk.commands.base import BaseCommand
from dftk.common.io import io


class PrintCommand(BaseCommand):
    """Read any dftk input and write TSV to stdout."""

    name = "print"
    help = "Write any input (TSV or parquet pipe) as TSV, JSON, Markdown or LaTeX"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # The standard output options (--move, --select, --removeheader, …)
        # work too; with -o FILE print writes there instead of stdout
        self.add_io_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        io.printdf(io.read(args), args)
