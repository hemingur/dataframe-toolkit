"""
dftk.commands.implode_cmd — dftk implode subcommand.

The inverse of explode: join the values of a column within each group of
rows into one cell ("GO:1,GO:2"). Several columns are joined in the same
row order, so explode -c a b followed by implode -c a b gives back the
original rows.
"""

import argparse

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

_EPILOG = """\
EXAMPLE
-------
  gene   go    tissue
  BRCA1  GO:1  breast
  BRCA1  GO:2  breast
  TP53   GO:3  colon

  dftk implode data.tsv -c go

  gene   go         tissue
  BRCA1  GO:1,GO:2  breast
  TP53   GO:3       colon

By default rows are grouped by all the other columns, so rows are only
joined when everything else in them is the same. With -i, rows are grouped
by those columns only and the other columns are dropped. Rows with an
empty key form a group of their own.

Values are joined in the order of the rows; --sortvalues sorts them and
--unique drops repeats. Empty values are left out, and a group with none
gets an empty cell. Whole numbers are written without ".0" when every
value in the column is one (pandas reads integers with gaps as floats).

Several columns are joined row by row, so their n-th items still belong
together; --unique and --sortvalues then work on whole rows.

Runnable version: each species' islands, one cell per species:

  dftk dataset penguins -o \\
      | dftk implode ... -c island -i species --unique --sortvalues

Numbers, with another separator: the party sizes seen on each day:

  dftk dataset tips -o \\
      | dftk implode ... -c size -i day --unique --sortvalues --sep '|'

Undoing explode: --poscol puts the items back in their original order,
here after shuffling the rows, and is dropped from the output:

  dftk dataset penguins -o \\
      | dftk implode ... -c island -i species --unique -o \\
      | dftk explode ... -c island --poscol n -o \\
      | dftk sample ... -f 1 --randomseed 1 -o \\
      | dftk implode ... -c island --poscol n
"""


def _as_text(s: pd.Series) -> pd.Series:
    """Return *s* as strings, keeping missing values missing."""
    values = s.dropna()
    if (
        pd.api.types.is_float_dtype(s)
        and len(values)
        and (values == values.round()).all()
    ):
        # Integers with gaps are read as floats: write 10, not 10.0
        s = s.astype("Int64")
    text = s.astype(object).where(s.notna())
    text = text.map(lambda v: v if pd.isna(v) else str(v))
    return text.where(text != "")


def _implode(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cols = args.cols
    poscol = args.poscol
    if len(set(cols)) != len(cols):
        raise ValueError(f"-c/--cols lists a column more than once: {cols}")
    if poscol is not None and poscol in cols:
        raise ValueError(f"--poscol {poscol!r} is also in -c/--cols")

    if args.indexcols is not None:
        keys = list(args.indexcols)
        overlap = [c for c in keys if c in cols or c == poscol]
        if overlap:
            raise ValueError(
                f"-i/--indexcols overlaps -c/--cols or --poscol: {overlap}"
            )
    else:
        keys = [c for c in df.columns if c not in cols and c != poscol]

    order = [c for c in df.columns if c in keys or c in cols]
    work = df[keys + cols + ([poscol] if poscol else [])].reset_index(drop=True)
    if keys:
        work["_gid"] = work.groupby(keys, sort=False, dropna=False).ngroup()
    else:
        work["_gid"] = 0

    if args.sortvalues:
        work = work.sort_values(["_gid", *cols], kind="stable", na_position="last")
    elif poscol:
        work = work.sort_values(["_gid", poscol], kind="stable")
    else:
        work = work.sort_values("_gid", kind="stable")
    if args.unique:
        work = work.drop_duplicates(subset=["_gid", *cols])

    for col in cols:
        work[col] = _as_text(work[col])
    grouped = work.groupby("_gid", sort=True)
    joined = grouped[cols].agg(lambda s: args.sep.join(s.dropna()))
    joined = joined.where(joined != "")
    result = grouped[keys].first().join(joined) if keys else joined
    return result.reset_index(drop=True)[order]


class ImplodeCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "implode"

    @property
    def help(self) -> str:
        return "Join a column's values within groups of rows into one cell"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("implode options")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            required=True,
            metavar="COL",
            help="Column(s) whose values are joined. Several columns are "
            "joined in the same row order.",
        )
        g.add_argument(
            "-i",
            "--indexcols",
            nargs="+",
            default=None,
            metavar="COL",
            help="Group by these columns and drop the others "
            "(default: group by all columns not in -c or --poscol).",
        )
        g.add_argument(
            "--sep",
            default=",",
            metavar="SEP",
            help="Separator between joined values (default: ,).",
        )
        g.add_argument(
            "--poscol",
            default=None,
            metavar="NAME",
            help="Join the values in the order of column NAME (e.g. the "
            "positions from explode --poscol) and drop it.",
        )
        g.add_argument(
            "--unique",
            action="store_true",
            help="Drop repeated values (with several -c columns, repeated rows).",
        )
        g.add_argument(
            "--sortvalues",
            action="store_true",
            help="Sort the values in each cell (with several -c columns, "
            "sort whole rows by them).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.cols, "-c/--cols")
        check_cols(df, args.indexcols, "-i/--indexcols")
        if args.poscol is not None:
            check_cols(df, [args.poscol], "--poscol")
        result = _implode(df, args)
        io.printdf(result, args)
