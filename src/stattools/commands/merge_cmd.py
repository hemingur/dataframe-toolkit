"""
stattools.commands.merge_cmd — dfstat merge subcommand.

Port of dfmerg.py, adapted for the BaseCommand pattern and the stattools
common.io module.

Key features
------------
* Inner, left, right, outer, and cross joins (pandas merge under the hood).
* Common-key merge (-k) or mapped-key merge (-lo / -ro).
* Duplicate column names automatically receive a ``_r`` suffix on the right
  side (or ``_l`` on the left when --only right is active).
* --only left/right  performs an outer join and returns only rows that appear
  exclusively on the specified side (anti-join).
* --select [:left:]  expands to all left-file columns, suppressing the ``_r``
  suffix on any duplicates.
* --select [:right:]  expands to all right-file columns, suppressing ``_l``.
* Both input files can be TSV or Parquet (.parquet extension auto-detected).
  The left file defaults to stdin, making it parquet-pipe friendly.
"""

import argparse
import logging
import uuid

import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import check_cols, io

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _set_suffixes(args: argparse.Namespace) -> None:
    """Configure merge suffix and indicator settings on *args*."""
    args.suffixes = ("", "_r")
    args.indicator = False

    if args.type == "cross":
        args.only = None  # cross join does not support --only

    if args.only is not None:
        args.type = "outer"
        args.indicator = True
        args.filtertag = f"{args.only}_only"
        if args.only == "right":
            args.suffixes = ("_l", "")


def _expand_select(
    left_cols: list[str],
    right_cols: list[str],
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Expand ``[:left:]`` / ``[:right:]`` tokens in ``args.select``.

    Mutates ``args.select`` in-place, replacing the token with the actual
    column names from that side.  Returns the ``(left_suffix, right_suffix)``
    tuple to use for the merge so that the expanded column names never carry
    an unwanted suffix.

    If no token is present, returns ``args.suffixes`` unchanged.
    """
    select: list[str] | None = getattr(args, "select", None)
    if select is None:
        return args.suffixes

    if "[:left:]" in select and "[:right:]" in select:
        raise ValueError("Cannot use both [:left:] and [:right:] in --select")

    try:
        idx = select.index("[:left:]")
        select[idx : idx + 1] = list(left_cols)
        return ("", "_r")
    except ValueError:
        pass

    try:
        idx = select.index("[:right:]")
        select[idx : idx + 1] = list(right_cols)
        return ("_l", "")
    except ValueError:
        pass

    return args.suffixes


def _add_cross_col(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Add a temporary UUID key column to both frames for a cross join."""
    colname = uuid.uuid1().hex
    df_left[colname] = 1
    df_right[colname] = 1
    args.type = "inner"
    args.keys = [colname]
    # Schedule the helper column for removal via io.printdf --drop
    drop = list(getattr(args, "drop", None) or [])
    drop.append(colname)
    args.drop = drop


def _filter_only(
    df: pd.DataFrame,
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """For ``--only`` mode: filter to rows exclusive to one side.

    The pandas ``_merge`` indicator column is removed.  If ``--select`` was
    not provided, the output is restricted to columns from the chosen side.
    """
    if args.only is None:
        return df

    df = df[df["_merge"] == args.filtertag].copy()
    df = df.drop(columns=["_merge"])

    # Default column selection: columns that came from the chosen side
    if getattr(args, "select", None) is None:
        if args.only == "left":
            args.select = [c for c in df_left.columns if c in df.columns]
        else:
            args.select = [c for c in df_right.columns if c in df.columns]

    return df


def _do_merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Perform the merge and return the combined DataFrame."""
    is_cross = args.type == "cross"
    if is_cross:
        _add_cross_col(df_left, df_right, args)
    elif args.keys is not None:
        check_cols(df_left, args.keys, "-k (left file)")
        check_cols(df_right, args.keys, "-k (right file)")
    elif args.left_on is not None or args.right_on is not None:
        check_cols(df_left, args.left_on, "-lo/--left_on")
        check_cols(df_right, args.right_on, "-ro/--right_on")

    suffixes = _expand_select(list(df_left.columns), list(df_right.columns), args)

    if args.keys is not None:
        df_merged = pd.merge(
            df_left,
            df_right,
            on=args.keys,
            how=args.type,
            indicator=args.indicator,
            suffixes=suffixes,
        )
    elif args.left_on is not None and args.right_on is not None:
        df_merged = pd.merge(
            df_left,
            df_right,
            left_on=args.left_on,
            right_on=args.right_on,
            how=args.type,
            indicator=args.indicator,
            suffixes=suffixes,
        )
    else:
        raise ValueError(
            "Specify -k/--keys for common columns, "
            "or both -lo and -ro for mapped columns."
        )

    df_merged = _filter_only(df_merged, df_left, df_right, args)

    # Safety: drop _merge indicator if somehow still present
    if "_merge" in df_merged.columns:
        df_merged = df_merged.drop(columns=["_merge"])

    return df_merged


# ---------------------------------------------------------------------------
# Command class
# ---------------------------------------------------------------------------

_EPILOG = """\
MERGE MODES
-----------
Common-key merge  (key column has the same name in both files):

  dfstat merge left.tsv -r right.tsv -k id
  cat left.tsv | dfstat merge -r right.tsv -k id -o | dfstat stat -c value

Mapped-key merge  (key columns have different names):

  dfstat merge left.tsv -r right.tsv -lo person_id -ro id


JOIN TYPES  (-t / --type)
--------------------------
  inner   (default) Only rows with matching keys in both files.
  left    All left rows; NaN where right has no match.
  right   All right rows; NaN where left has no match.
  outer   All rows from both files; NaN where no match.
  cross   Cartesian product (no keys required or allowed).


ANTI-JOIN  (--only)
--------------------
Return rows that appear exclusively in one file:

  dfstat merge left.tsv -r right.tsv -k id --only left
  dfstat merge left.tsv -r right.tsv -k id --only right

Internally this runs an outer join and filters rows where the pandas
_merge indicator equals "left_only" or "right_only".


COLUMN SELECTION FROM ONE SIDE  (--select)
------------------------------------------
When both files share column names beyond the key, the right-side copy
receives a ``_r`` suffix by default.  To output only one side's columns
without any suffix pollution, use the special tokens:

  --select [:left:]   All left-file columns (no _r suffix needed).
  --select [:right:]  All right-file columns (no _l suffix needed).

Tokens can be mixed with explicit names:

  --select id score [:left:]

These tokens are expanded before the merge, so the resulting column list
is what you see in the output.


PARQUET INPUT AND SELF-MERGE
-----------------------------
Both files accept .parquet paths; the extension is detected automatically.
The left file defaults to stdin, making it parquet-pipe friendly.

Self-merge — join a dataframe with itself — is supported with -r - :

  cat data.tsv | dfstat eval "y = sqrt(x)" -o | dfstat merge -l - -r - -k x

When both -l and -r are -, stdin is read once and df_right is a copy of
df_left.  This works naturally with the parquet-pipe: the previous command
writes a temp parquet and prints its path; dfstat merge reads that path,
loads the file, and reuses it for both sides.

Named-file self-merge (no stdin required):

  dfstat merge data.tsv -r data.tsv -k id --only left
"""


class MergeCommand(BaseCommand):
    name = "merge"
    help = "Join two dataframes on common or mapped keys."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        # Left file via the standard DATAFILE positional (stdin-friendly).
        io.parser_read(parser)
        io.parser_output(parser)

        g = parser.add_argument_group("merge options")
        g.add_argument(
            "-l",
            "--leftfile",
            default=None,
            metavar="FILE",
            help=(
                "Left-hand input file (TSV or .parquet), or - for stdin.  "
                "Overrides the positional DATAFILE argument when provided."
            ),
        )
        g.add_argument(
            "-r",
            "--rightfile",
            default=None,
            metavar="FILE",
            help=(
                "Right-hand input file (TSV or .parquet), or - for stdin.  "
                "When - is used and the left file is also stdin, the left "
                "DataFrame is reused as-is (self-join / self-merge)."
            ),
        )
        g.add_argument(
            "-k",
            "--keys",
            nargs="+",
            default=None,
            metavar="COL",
            help="Column name(s) shared by both files on which to merge.",
        )
        g.add_argument(
            "-lo",
            "--left_on",
            nargs="+",
            default=None,
            metavar="COL",
            help="Left-file key column(s) when names differ from the right file.",
        )
        g.add_argument(
            "-ro",
            "--right_on",
            nargs="+",
            default=None,
            metavar="COL",
            help="Right-file key column(s) when names differ from the left file.",
        )
        g.add_argument(
            "-t",
            "--type",
            choices=["inner", "left", "right", "outer", "cross"],
            default="inner",
            help="Join type (default: inner).",
        )
        g.add_argument(
            "--only",
            choices=["left", "right"],
            default=None,
            help=(
                "Return rows exclusive to one side (anti-join).  Implies --type outer."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.rightfile is None:
            raise ValueError("Specify -r FILE (or -r - for self-join).")

        # -l overrides the positional DATAFILE when provided
        if args.leftfile is not None:
            args.DATAFILE = args.leftfile

        left_was_stdin = args.DATAFILE in (None, "-")

        # Read left file (positional DATAFILE, -l, or stdin)
        try:
            df_left = io.read(args)
        except Exception:
            logger.error("Failed to read left (primary) input file")
            raise

        _set_suffixes(args)

        # Self-join: -r - reuses the left DataFrame (stdin already consumed)
        if args.rightfile == "-":
            if not left_was_stdin:
                logger.warning(
                    "-r - used but left file was not stdin; "
                    "treating as self-join (copy of left)."
                )
            df_right = df_left.copy()
        else:
            args.DATAFILE = args.rightfile
            try:
                df_right = io.read(args)
            except Exception:
                logger.error(f"Failed to read right file: {args.rightfile!r}")
                raise

        df_merged = _do_merge(df_left, df_right, args)
        io.printdf(df_merged, args)
