"""
dftk.commands.sample_cmd — dftk sample subcommand.

Draw a random sample of rows from a DataFrame, optionally within groups.

Sampling size (one required)
-----------------------------
  -n N      draw exactly N rows
  -f FRAC   draw a fraction of rows (0.0 – 1.0); use 1.0 to shuffle

Options
-------
  -g COL …     sample independently within each group
  --randomseed random seed — integer or string (strings are MD5-hashed)
  --replace    sample with replacement

Examples
--------
    dftk sample data.tsv -n 100
    dftk sample data.tsv -f 0.8
    dftk sample data.tsv -n 50 -g treatment --randomseed 42
    dftk sample data.tsv -f 1.0 --randomseed shuffle
"""

import argparse
import sys

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import io
from dftk.common.seed import normalize_seed


class SampleCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def help(self) -> str:
        return "Draw a random sample of rows, optionally within groups"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap

        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.epilog = """\
EXAMPLES
--------
  dftk sample data.tsv -n 100
  dftk sample data.tsv -f 0.8
  dftk sample data.tsv -n 50 -g treatment --randomseed 42
  dftk sample data.tsv -f 1.0 --randomseed shuffle   # reproducible shuffle
  dftk sample data.tsv -n 10 -g group --replace
"""
        io.parser_read(parser)

        g = parser.add_argument_group("sampling")
        size = g.add_mutually_exclusive_group(required=True)
        size.add_argument(
            "-n",
            dest="samplesize",
            type=int,
            default=None,
            metavar="N",
            help="Number of rows to sample.",
        )
        size.add_argument(
            "-f",
            dest="samplefrac",
            type=float,
            default=None,
            metavar="FRAC",
            help="Fraction of rows to sample (0.0 – 1.0).  Use 1.0 to shuffle.",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="Sample independently within each group defined by these columns.",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help=(
                "Random seed for reproducibility.  "
                "Accepts an integer or any string (hashed via MD5 to a uint32)."
            ),
        )
        g.add_argument(
            "--replace",
            action="store_true",
            help="Sample with replacement.",
        )

        io.parser_output(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)

        seed = normalize_seed(args.randomseed)
        if seed is not None:
            print(f"randomseed = {seed}", file=sys.stderr)

        rng = np.random.default_rng(seed)

        sample_kwargs: dict = {"replace": args.replace, "random_state": rng}
        if args.samplesize is not None:
            sample_kwargs["n"] = args.samplesize
        else:
            sample_kwargs["frac"] = args.samplefrac

        if args.groupcol is not None:
            pieces = [
                grp.sample(**sample_kwargs) for _, grp in df.groupby(args.groupcol)
            ]
            result = pd.concat(pieces).reset_index(drop=True)
        else:
            result = df.sample(**sample_kwargs).reset_index(drop=True)

        io.printdf(result, args)
