"""
stattools.commands.pivot_cmd — dfstat pivot subcommand.

Wraps pandas pivot_table with optional bootstrap resampling and three
sampling strategies for hierarchical data.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io
from stattools.common.seed import normalize_seed

logger = logging.getLogger(__name__)

_EPILOG = """\
AGGREGATION FUNCTIONS (-f / --aggfunc)
---------------------------------------
Pass one or more names from this list:

  count  min  max  mean  median  mode  sum  std  var  skew  kurt  prod
  cilo   cihi   (confidence-interval bounds, level set by --confidencelevel)
  bitwise_and  bitwise_or  bitwise_xor

Bare names apply the same function(s) to every value column:

  dfstat pivot data.tsv -v sales cost -i region -f mean std

Per-column functions use COL:FUNC syntax — all items must use this form if
any do:

  dfstat pivot data.tsv -v sales cost -i region -f sales:mean cost:sum

When -f is omitted, pandas default aggregation (mean) is used.


BOOTSTRAP SAMPLING MODES (--bootstrap N)
-----------------------------------------
Three sampling strategies are available, determined by which flags are given.

  1. Full resampling (--fullsampling)
     Sample every row in the dataset with replacement, ignoring group
     structure.  Use this when there is no meaningful grouping to preserve.

       dfstat pivot data.tsv -v value -i subject -f mean \\
         --bootstrap 500 --fullsampling

  2. Group-based row sampling (--samplinggroup COLS)
     Resample rows with replacement *within* each group defined by
     COLS.  Group sizes are preserved across samples.

       dfstat pivot data.tsv -v value -i timepoint -g condition -f mean \\
         --bootstrap 500 --samplinggroup subject

     If neither --fullsampling nor --samplinggroup nor --samplingcols is
     given, the sampling groups default to the union of --index and
     --groupcols (the pivot dimensions themselves).

  3. Hierarchical (parent/child) sampling (--samplingcols COLS)
     Use when each logical unit has multiple data rows (e.g. a subject has
     many measurements).  Sampling parents and pulling all their children
     keeps the within-unit structure intact.

     Steps:
       a. Build a sampling set of unique combinations of --samplingcols
          (the parent identifiers).
       b. Bootstrap-resample that set with replacement — optionally within
          groups defined by --samplinggroup.
       c. Merge the resampled set back against the full DataFrame, pulling
          in all child rows for each selected parent.

     If parent P is drawn twice, all of P's rows appear twice in the
     bootstrap sample; if Q is not drawn, none of Q's rows appear.

       # Sample subjects (parents) within each sex group;
       # each subject has multiple visit rows (children).
       dfstat pivot data.tsv -v score -i visit -g condition -f mean \\
         --bootstrap 500 \\
         --samplingcols subject_id \\
         --samplinggroup sex

     Note: --samplingcols values are combined with --samplinggroup values
     to form the merge key, so the group column must also be in the data.


PIPE MODE
---------
Use -o alone to pass data to the next dfstat command without writing to a
file:

  cat data.tsv \\
    | dfstat pivot - -v value -i subject -g condition -f mean -o \\
    | dfstat stat - -c condition_A condition_B
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ci(name: str, level: float, method: str):
    """Return a named percentile function for use as a pandas agg function."""

    def cifunction(a):
        return np.percentile(a, q=level, method=method)

    cifunction.__name__ = name
    return cifunction


def _bitwise(op: str):
    """Return a named bitwise-reduce function."""

    def bitfunction(a):
        if op == "and":
            return np.bitwise_and.reduce(a)
        if op == "or":
            return np.bitwise_or.reduce(a)
        if op == "xor":
            return np.bitwise_xor.reduce(a)

    bitfunction.__name__ = op
    return bitfunction


def _prepare_funcs(
    aggfuncs: list[str] | None, confidencelevel: float, confidencemethod: str
):
    """Convert aggfunc name strings into a pandas-compatible aggfunc argument.

    Returns:
        None          → pandas default (mean)
        list          → same function(s) applied to all value columns
        dict          → per-column functions, keyed by column name
    """
    if aggfuncs is None:
        return None

    funclist: list = []
    funcdict: dict = {}

    for item in aggfuncs:
        colname = None
        funcname = item
        if ":" in item:
            colname, funcname = item.split(":", 1)
            funcdict.setdefault(colname, [])

        if funcname in {
            "count",
            "min",
            "max",
            "mean",
            "median",
            "mode",
            "sum",
            "std",
            "var",
            "skew",
            "kurt",
            "prod",
        }:
            func = funcname
        elif funcname == "cilo":
            lo = (100.0 - confidencelevel) / 2.0
            func = _ci("cilo", lo, confidencemethod)
        elif funcname == "cihi":
            hi = 100.0 - (100.0 - confidencelevel) / 2.0
            func = _ci("cihi", hi, confidencemethod)
        elif funcname in {"bitwise_and", "bitwise_or", "bitwise_xor"}:
            func = _bitwise(funcname.split("_", 1)[1])
        else:
            try:
                func = getattr(np, funcname)
            except AttributeError:
                raise ValueError(  # noqa: E501
                    f"Unknown aggregation function: {funcname!r}"
                ) from None

        if funcdict and colname is None:
            raise ValueError(
                "Mixed aggfunc spec: either all items must use COL:FUNC "
                "syntax or none must."
            )

        if colname is not None:
            funcdict[colname].append(func)
        else:
            funclist.append(func)

    return funcdict if funcdict else funclist


def _flatten_columns(cols) -> list[str]:
    """Flatten MultiIndex column tuples to underscore-joined strings."""
    result = []
    for col in cols:
        if isinstance(col, str):
            result.append(col)
        else:
            result.append("_".join(str(x) for x in col if str(x) != ""))
    return result


def _do_pivot(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Run pivot_table and flatten column names."""
    funcs = _prepare_funcs(
        getattr(args, "aggfunc", None),
        getattr(args, "confidencelevel", 95.0),
        getattr(args, "confidencemethod", "linear"),
    )

    index = args.index or None
    columns = args.groupcols or None

    pivoted = df.pivot_table(
        values=args.values,
        index=index,
        columns=columns,
        aggfunc=funcs if funcs is not None else "mean",
    )

    cols = pivoted.columns
    pivoted.columns = _flatten_columns(cols.values if hasattr(cols, "values") else cols)

    if getattr(args, "fillzero", False) and hasattr(pivoted.index, "levels"):
        pidx = pd.MultiIndex.from_product(
            pivoted.index.levels, names=pivoted.index.names
        )
        pivoted = pivoted.reindex(pidx, fill_value=0, copy=False)

    return pivoted.reset_index()


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class PivotCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "pivot"

    @property
    def help(self) -> str:
        return "Pivot / aggregate a DataFrame (pivot_table)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap

        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        io.parser_read(parser)

        g = parser.add_argument_group("pivot options")
        g.add_argument(
            "-v",
            "--values",
            help="Value column(s) to aggregate.",
            nargs="+",
            required=True,
            metavar="COL",
        )
        g.add_argument(
            "-i",
            "--index",
            help="Column(s) that become row labels in the pivot table.",
            nargs="*",
            default=[],
            metavar="COL",
        )
        g.add_argument(
            "-g",
            "--groupcols",
            help="Column(s) whose unique values are spread into new column headers.",
            nargs="*",
            default=[],
            metavar="COL",
        )
        g.add_argument(
            "-f",
            "--aggfunc",
            help=(
                "Aggregation function(s).  "
                "Use bare names (mean std) or COL:FUNC pairs (sales:mean cost:sum).  "
                "See --help for full list.  Default: mean."
            ),
            nargs="*",
            default=None,
            metavar="FUNC",
        )
        g.add_argument(
            "-z",
            "--fillzero",
            help="Reindex to all (index × columns) combinations, fill missing with 0.",
            action="store_true",
        )
        g.add_argument(
            "--confidencelevel",
            help="Confidence level for cilo/cihi aggfuncs, in percent (default: 95.0).",
            type=float,
            default=95.0,
            metavar="PCT",
        )
        g.add_argument(
            "--confidencemethod",
            help="Interpolation method for cilo/cihi (default: linear).",
            default="linear",
            metavar="METHOD",
        )

        bs = parser.add_argument_group("bootstrap options")
        bs.add_argument(
            "--bootstrap",
            help=(
                "Number of bootstrap samples.  "
                "Produces a single DataFrame with a samplenum column."
            ),
            type=int,
            default=None,
            metavar="N",
        )
        bs.add_argument(
            "--randomseed",
            help="Random seed.  Accepts an integer or a string (hashed to uint32).",
            default=None,
            metavar="SEED",
        )
        bs.add_argument(
            "--fullsampling",
            help="Resample all rows with replacement, ignoring group structure.",
            action="store_true",
        )
        bs.add_argument(
            "--samplinggroup",
            help=(
                "Column(s) defining groups within which rows are resampled.  "
                "Also used as the within-group dimension when --samplingcols is given."
            ),
            nargs="*",
            default=[],
            metavar="COL",
        )
        bs.add_argument(
            "--samplingcols",
            help=(
                "Column(s) identifying the sampling unit (e.g. a subject ID).  "
                "Enables hierarchical bootstrap: unique units are resampled and "
                "all their rows are pulled in via merge.  "
                "See --help for a detailed example."
            ),
            nargs="*",
            default=[],
            metavar="COL",
        )

        io.parser_output(parser)

    def execute(self, args: argparse.Namespace) -> None:
        args.randomseed = normalize_seed(args.randomseed)
        np.random.seed(args.randomseed)

        df = io.read(args)

        check_cols(df, args.values, "-v/--values")
        check_cols(df, args.index, "-i/--index")
        check_cols(df, args.groupcols, "-g/--groupcols")
        check_cols(df, args.samplingcols, "--samplingcols")
        check_cols(df, args.samplinggroup, "--samplinggroup")

        if args.bootstrap is None:
            result = _do_pivot(df, args)
            io.printdf(result, args)
            return

        # ------------------------------------------------------------------
        # Bootstrap
        # ------------------------------------------------------------------

        logger.info(
            f"Bootstrap pivot: {args.bootstrap} samples, seed={args.randomseed}"
        )

        # Determine sampling mode and build sampling set if needed.
        if args.fullsampling:
            mode = "full"
            s = None
            samplinggroup: list[str] = []
        elif args.samplingcols:
            mode = "hierarchical"
            # The merge key is samplingcols + samplinggroup so the group
            # column is preserved through the merge.
            merge_cols = args.samplingcols + args.samplinggroup
            s = df[merge_cols].drop_duplicates().reset_index(drop=True)
            samplinggroup = args.samplinggroup
        else:
            mode = "grouped"
            s = None
            samplinggroup = args.samplinggroup or (args.index + args.groupcols)

        logger.info(f"Sampling mode: {mode}")

        frames: list[pd.DataFrame] = []

        for ns in range(1, args.bootstrap + 1):
            if mode == "full":
                bf = df.sample(frac=1, replace=True)

            elif mode == "hierarchical":
                if samplinggroup:
                    ss = (
                        s.groupby(samplinggroup, group_keys=False)[s.columns.tolist()]
                        .apply(lambda x: x.sample(frac=1.0, replace=True))
                        .reset_index(drop=True)
                    )
                else:
                    ss = s.sample(frac=1.0, replace=True).reset_index(drop=True)
                bf = pd.merge(ss, df)

            else:  # grouped
                bf = (
                    df.groupby(samplinggroup, group_keys=False)[df.columns.tolist()]
                    .apply(lambda x: x.sample(frac=1.0, replace=True))
                    .reset_index(drop=True)
                )

            result = _do_pivot(bf, args)
            result["samplenum"] = ns
            frames.append(result)

        io.printdf(pd.concat(frames, ignore_index=True), args)
