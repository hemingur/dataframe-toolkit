"""
dftk.commands.split_cmd — dftk split subcommand.

Port of dfsplit.py: split a dataframe into one TSV file per group.
"""

import argparse
import copy
import os

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io


def _make_filename(groupname, prefix: str, suffix: str) -> str:
    tag = (
        "_".join(str(x) for x in groupname)
        if isinstance(groupname, tuple)
        else str(groupname)
    )
    return f"{prefix}{tag}{suffix}"


class _SafeDict(dict):
    """Format-map dict that leaves unknown keys unreplaced."""

    def __missing__(self, key):
        return "{" + key + "}"


def _split(df, args: argparse.Namespace) -> None:
    for groupname, groupdf in df.groupby(args.groups):
        if args.template is not None:
            key = (groupname,) if not isinstance(groupname, tuple) else groupname
            group_dict = dict(zip(args.groups, key, strict=False))
            filename = args.template.format_map(_SafeDict(group_dict))
        else:
            filename = _make_filename(groupname, args.prefix, args.suffix)

        if args.noclobber and os.path.exists(filename):
            continue

        file_args = copy.copy(args)
        file_args.output = filename
        io.printdf(groupdf, file_args)


_EPILOG = """\
FILENAME PATTERNS
-----------------
  By default output files are named:  <prefix><group_key><suffix>

    dftk split data.tsv -g country -p out/ -s .tsv
    # → out/Iceland.tsv, out/Norway.tsv, ...

  With --template, use {col} placeholders for the group column values:

    dftk split data.tsv -g year month --template data_{year}_{month}.tsv

  Unknown placeholders are left as-is (safe format_map).

OPTIONS
-------
  --noclobber   Skip files that already exist instead of overwriting.

  Note: -o/--output is not used by split; output paths are always determined
  by -p/--prefix, -s/--suffix, or --template.

EXAMPLES
--------
  dftk split data.tsv -g group
  dftk split data.tsv -g country year -p results/ -s .tsv
  dftk split data.tsv -g category --template out/{category}.tsv
"""


class SplitCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "split"

    @property
    def help(self) -> str:
        return "Split a dataframe into one file per group."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("split options")
        g.add_argument(
            "-g",
            "--groups",
            nargs="+",
            required=True,
            metavar="COL",
            help="Column(s) to group and split by.",
        )
        g.add_argument(
            "-p",
            "--prefix",
            default="split-",
            metavar="PREFIX",
            help="Filename prefix (default: 'split-').",
        )
        g.add_argument(
            "-s",
            "--suffix",
            default="",
            metavar="SUFFIX",
            help="Filename suffix / extension (default: none).",
        )
        g.add_argument(
            "--template",
            default=None,
            metavar="TEMPLATE",
            help="Filename template with {col} placeholders, e.g. out/{group}.tsv.",
        )
        g.add_argument(
            "--noclobber",
            action="store_true",
            help="Skip output files that already exist.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.groups, "-g/--groups")
        _split(df, args)
