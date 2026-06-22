"""
stattools.commands.randvar_cmd — dfstat randvar subcommand.

Port of dfrvs.py: append a column of random variates drawn from any
scipy.stats distribution, or generate a stand-alone sample without an
input file.

Usage
-----
  # Append a column to an existing file
  dfstat randvar data.tsv -d x --dist norm --parameters loc:0,scale:1

  # Generate a stand-alone sample (no input file)
  dfstat randvar -n 1000 -d x --dist norm --parameters loc:0,scale:1
  dfstat randvar -n 500  -d x --dist beta --parameters a:2,b:5 --randomseed 42

  # List all available distributions and their parameters
  dfstat randvar --list
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import scipy.stats as ss

from stattools.commands.base import BaseCommand
from stattools.common.io import io
from stattools.common.seed import normalize_seed

logger = logging.getLogger(__name__)

_SCIPY_DOCS = "https://docs.scipy.org/doc/scipy/reference/stats.html"

_EPILOG = f"""\
PARAMETERS
----------
Distribution parameters are passed as key:val pairs separated by commas:

  --parameters loc:0,scale:1
  --parameters a:2,b:5,loc:0,scale:1

The parameter names are distribution-specific.  Use --list to see the
accepted parameters for every distribution, or consult the scipy.stats
documentation:

  {_SCIPY_DOCS}

RANDOM SEED
-----------
--randomseed accepts an integer or any string.  A non-numeric string is
hashed (MD5) to a reproducible uint32 seed, which is printed to stderr.

EXAMPLES
--------
  dfstat randvar data.tsv -d noise --dist norm
  dfstat randvar data.tsv -d noise --dist norm --parameters loc:5,scale:2
  dfstat randvar -n 1000 -d u --dist uniform --randomseed 99
  dfstat randvar -n 500  -d k --dist poisson --parameters mu:3
  dfstat randvar --list
  dfstat randvar --list | dfstat query - --query "type == 'continuous'"
"""


def _parse_parameters(paramstring: str | None) -> dict:
    """Parse 'key:val,key:val' into {key: float} dict."""
    if not paramstring:
        return {}
    pairs = [x.split(":") for x in paramstring.split(",")]
    return {k: float(v) for k, v in pairs}


def _list_distributions() -> pd.DataFrame:
    """Return a DataFrame of all scipy.stats distributions and their parameters."""
    rows = []
    for name in sorted(ss._continuous_distns._distn_names):
        dist = getattr(ss, name)
        shapes = dist.shapes or ""
        params = (shapes + ", " if shapes else "") + "loc, scale"
        rows.append({"name": name, "type": "continuous", "parameters": params})
    for name in sorted(ss._discrete_distns._distn_names):
        dist = getattr(ss, name)
        shapes = dist.shapes or ""
        params = (shapes + ", " if shapes else "") + "loc"
        rows.append({"name": name, "type": "discrete", "parameters": params})
    return pd.DataFrame(rows)


class RandvarCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "randvar"

    @property
    def help(self) -> str:
        return "Append a column of random variates from a scipy.stats distribution."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("distribution")
        g.add_argument(
            "-d",
            "--destcol",
            default=None,
            metavar="COL",
            help="Name of the new column to append.",
        )
        g.add_argument(
            "--dist",
            default=None,
            metavar="DIST",
            help="scipy.stats distribution name (e.g. norm, uniform, beta, poisson).",
        )
        g.add_argument(
            "--parameters",
            default=None,
            metavar="KEY:VAL,...",
            help="Distribution parameters as key:val pairs, e.g. loc:0,scale:1.",
        )
        g.add_argument(
            "--randomseed",
            default=None,
            metavar="SEED",
            help="Random seed (integer or string; strings are hashed to uint32).",
        )
        g.add_argument(
            "-n",
            "--nsamples",
            type=int,
            default=None,
            metavar="N",
            help=(
                "Number of samples to generate.  Required when no DATAFILE is "
                "provided and stdin is not piped; ignored otherwise."
            ),
        )
        g.add_argument(
            "--list",
            action="store_true",
            help="List all available scipy.stats distributions and their parameters.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        # --list short-circuits everything
        if args.list:
            df = _list_distributions()
            io.printdf(df, args)
            return

        # Validate required args (not enforced at parse time to allow --list)
        if not args.destcol:
            logger.error("-d/--destcol is required.")
            sys.exit(1)
        if not args.dist:
            logger.error("--dist is required.")
            sys.exit(1)

        # Resolve distribution
        try:
            dist_cls = getattr(ss, args.dist)
        except AttributeError:
            logger.error(
                "Distribution %r not found in scipy.stats. Use --list.",
                args.dist,
            )
            sys.exit(1)

        kwargs = _parse_parameters(args.parameters)
        try:
            rv = dist_cls(**kwargs)
        except TypeError as exc:
            logger.error("Invalid parameters for %r: %s", args.dist, exc)
            sys.exit(1)

        seed = normalize_seed(args.randomseed)
        rng = np.random.default_rng(seed)

        # -n given → standalone mode (no input file needed)
        # -n absent → read DATAFILE or stdin, use len(df) as sample count
        if args.nsamples is not None:
            df = pd.DataFrame(
                {args.destcol: rv.rvs(size=args.nsamples, random_state=rng)}
            )
        else:
            if args.DATAFILE is None and sys.stdin.isatty():
                logger.error(
                    "No input: provide a DATAFILE, pipe to stdin, or use -n/--nsamples."
                )
                sys.exit(1)
            df = io.read(args)
            df[args.destcol] = rv.rvs(size=len(df), random_state=rng)

        io.printdf(df, args)
