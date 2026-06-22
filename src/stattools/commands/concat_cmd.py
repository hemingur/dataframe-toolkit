"""
stattools.commands.concat_cmd — dfstat concat subcommand.

Concatenate two or more input files (TSV or parquet) row-wise into a single
DataFrame.  Column alignment follows pandas outer-join semantics: columns
present in some but not all files are filled with NaN (or --fill value).

Examples
--------
    dfstat concat a.tsv b.tsv c.tsv
    dfstat concat a.tsv b.tsv --sourcecol source
    dfstat concat a.parquet b.parquet -o
    dfstat eval data.tsv -f "x=1" -o | dfstat concat ... extra.tsv
"""

import argparse
import copy

import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io


class ConcatCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "concat"

    @property
    def help(self) -> str:
        return "Concatenate multiple input files row-wise"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap

        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.epilog = """\
COLUMN ALIGNMENT
----------------
  Columns present in all files are aligned by name.
  Columns missing from some files are filled with NaN (or --fill value).
  Use --select to enforce a fixed output column set.

EXAMPLES
--------
  dfstat concat a.tsv b.tsv
  dfstat concat a.tsv b.tsv c.tsv --sourcecol source
  dfstat concat a.parquet b.parquet -o result.parquet
  dfstat eval data.tsv -f "x=1" -o | dfstat concat ... extra.tsv
"""
        g = parser.add_argument_group("concat options")
        g.add_argument(
            "DATAFILES",
            nargs="+",
            metavar="FILE",
            help=(
                "Input files to concatenate (TSV or .parquet).  "
                "Use - for TSV stdin or ... for a parquet pipe path on stdin.  "
                "At most one stdin source is allowed."
            ),
        )
        g.add_argument(
            "--sourcecol",
            default=None,
            metavar="COL",
            help="Add a column with this name containing the source filename.",
        )
        g.add_argument(
            "--fill",
            default=None,
            metavar="VALUE",
            help=(
                "Fill value for columns absent in some files "
                "(default: NaN).  Use 0 or '' for numeric/string defaults."
            ),
        )
        g.add_argument(
            "--noheader",
            help="Input files have no header row; columns are named V1, V2, …",
            action="store_true",
        )
        g.add_argument(
            "--delimiter",
            help="Column delimiter for TSV inputs (default: tab)",
            default=None,
        )

        io.parser_output(parser)

    def execute(self, args: argparse.Namespace) -> None:
        stdin_used = False
        dfs: list[pd.DataFrame] = []

        for path in args.DATAFILES:
            if path in ("-", "..."):
                if stdin_used:
                    raise ValueError(
                        "stdin (-/...) can only appear once in a concat file list"
                    )
                stdin_used = True

            # Build a per-file args with DATAFILE set to this path.
            file_args = copy.copy(args)
            file_args.DATAFILE = path
            file_args.prequery = []

            try:
                df = io.read(file_args)
            except pd.errors.EmptyDataError:
                import logging

                logging.warning(f"concat: skipping empty file {path!r}")
                continue

            if args.sourcecol:
                df[args.sourcecol] = path

            dfs.append(df)

        if not dfs:
            raise ValueError("concat: no data read from any input file")

        result = pd.concat(dfs, ignore_index=True)

        if args.fill is not None:
            fill: int | float | str
            try:
                fill = int(args.fill)
            except ValueError:
                try:
                    fill = float(args.fill)
                except ValueError:
                    fill = args.fill
            result = result.fillna(fill)

        io.printdf(result, args)
