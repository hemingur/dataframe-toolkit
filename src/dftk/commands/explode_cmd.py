"""
dftk.commands.explode_cmd — dftk explode subcommand.

Split a column of separated values (e.g. "GO:1, GO:2") into one row per
value, repeating the other columns on each new row (pandas str.split +
explode). Several columns are split in parallel: their n-th items end up
on the same row, so every row must have the same number of items in them.
"""

import argparse

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

_EPILOG = """\
EXAMPLE
-------
  gene   go          tissue
  BRCA1  GO:1, GO:2  breast
  TP53   GO:3        colon

  dftk explode data.tsv -c go

  gene   go    tissue
  BRCA1  GO:1  breast
  BRCA1  GO:2  breast
  TP53   GO:3  colon

Spaces around each item are removed. An empty cell stays as one row with
an empty value; --dropempty removes such rows. If all the new values in a
split column are numbers, the column becomes numeric (integers stay
integers); columns named in --readasobject stay text.

Several columns are split in parallel, item by item. For every
combination instead, run explode once per column:

  id  a    b
  1   x,y  10,20

  dftk explode data.tsv -c a b       ->  (1, x, 10), (1, y, 20)

Runnable version: collect each species' islands into one comma-separated
cell with SQL, then split them back into rows, numbering each item:

  dftk dataset penguins -o \\
      | dftk query ... -o --sql "SELECT species,
            string_agg(DISTINCT island, ',' ORDER BY island) AS islands
            FROM data GROUP BY species ORDER BY species" \\
      | dftk explode ... -c islands --poscol n

Another separator, keeping only some columns:

  dftk dataset tips -o \\
      | dftk query ... -o --sql "SELECT day,
            string_agg(DISTINCT size::VARCHAR, '|' ORDER BY size::VARCHAR) AS sizes
            FROM data GROUP BY day ORDER BY day" \\
      | dftk explode ... -c sizes --sep '|' --select sizes day
"""


def _split_cell(value, sep: str):
    if pd.isna(value):
        return value
    return [item.strip() for item in str(value).split(sep)]


def _to_numeric(s: pd.Series) -> pd.Series:
    """Return *s* as integers or floats if every non-empty value is one."""
    values = s.dropna()
    if values.empty:
        return s
    text = values.astype(str)
    if text.str.fullmatch(r"[+-]?\d+").all():
        return s.astype("Int64") if s.isna().any() else s.astype("int64")
    try:
        return pd.to_numeric(s)
    except (ValueError, TypeError):
        return s


def _explode(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cols = args.cols
    if not args.sep:
        raise ValueError("--sep must not be empty")
    if len(set(cols)) != len(cols):
        raise ValueError(f"-c/--cols lists a column more than once: {cols}")
    if args.poscol is not None and args.poscol in df.columns:
        raise ValueError(f"--poscol {args.poscol!r} is already a column")

    df = df.reset_index(drop=True)
    for col in cols:
        df[col] = df[col].map(lambda v: _split_cell(v, args.sep)).astype(object)

    if len(cols) > 1:
        counts = pd.DataFrame(
            {c: df[c].map(lambda v: len(v) if isinstance(v, list) else 1) for c in cols}
        )
        bad = counts.index[counts.nunique(axis=1) > 1]
        if len(bad):
            shown = "; ".join(
                f"row {i + 1} ("
                + ", ".join(f"{c}: {counts.at[i, c]}" for c in cols)
                + ")"
                for i in bad[:5]
            )
            more = f" and {len(bad) - 5} more" if len(bad) > 5 else ""
            raise ValueError(
                f"Rows with different numbers of items in -c/--cols: {shown}{more}. "
                "Run explode once per column for every combination."
            )

    result = df.explode(cols)
    if args.poscol is not None:
        result[args.poscol] = result.groupby(level=0).cumcount() + 1
    result = result.reset_index(drop=True)

    for col in cols:
        result[col] = result[col].replace("", pd.NA)
    if args.dropempty:
        result = result[result[cols].notna().any(axis=1)].reset_index(drop=True)

    readasobject = getattr(args, "readasobject", None)
    for col in cols:
        if readasobject is not None and (not readasobject or col in readasobject):
            continue
        result[col] = _to_numeric(result[col])
    return result


class ExplodeCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "explode"

    @property
    def help(self) -> str:
        return "Split separated values in a column into one row per value"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("explode options")
        g.add_argument(
            "-c",
            "--cols",
            nargs="+",
            required=True,
            metavar="COL",
            help="Column(s) to split. Several columns are split in parallel "
            "and must have the same number of items on each row.",
        )
        g.add_argument(
            "--sep",
            default=",",
            metavar="SEP",
            help="Separator between items, a plain string, not a regex (default: ,).",
        )
        g.add_argument(
            "--poscol",
            default=None,
            metavar="NAME",
            help="Add a column NAME with each item's position in its cell (1, 2, ...).",
        )
        g.add_argument(
            "--dropempty",
            action="store_true",
            help="Drop rows whose split columns are all empty "
            "(default: keep them, so no input row is lost).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.cols, "-c/--cols")
        result = _explode(df, args)
        io.printdf(result, args)
