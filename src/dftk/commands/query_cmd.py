"""
dftk.commands.query_cmd — dftk query subcommand.

Filter (and optionally reshape) a dataframe using either pandas query
expressions or full DuckDB SQL.

Two modes
---------
pandas mode  (-q / --query, repeatable)
    Applies pandas DataFrame.query() expressions in sequence.  Each
    expression is a boolean row filter; all expressions are ANDed together.

      dftk query data.tsv -q "x > 0" -q "group == 'A'"

sql mode  (--sql)
    Passes a full SQL statement to DuckDB.  The input data is available as
    a virtual table named "data" (override with --table).  Any SQL that
    DuckDB supports is valid: SELECT, aggregation, window functions, CTEs.

      dftk query data.tsv --sql "SELECT grp, AVG(x) AS mean FROM data GROUP BY grp"

    When the input is a named .parquet file, DuckDB queries it natively
    without loading the full file into memory — suitable for large datasets.

    Requires: pip install duckdb
"""

import argparse
import json
import logging
import os

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common import docformats
from dftk.common.io import io

try:
    import duckdb

    _HAS_DUCKDB = True
except ImportError:
    duckdb = None
    _HAS_DUCKDB = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def _pandas_query(df: pd.DataFrame, expressions: list[str]) -> pd.DataFrame:
    """Apply each expression as a pandas boolean row filter (AND logic)."""
    for expr in expressions:
        try:
            df = df.query(expr)
        except Exception as exc:
            raise ValueError(f"pandas query failed ({expr!r}): {exc}") from exc
    return df


def _sql_query(args: argparse.Namespace, sql: str, table: str) -> pd.DataFrame:
    """Run *sql* against the input data registered as a DuckDB virtual table.

    For named .parquet inputs the file is queried as a lazy DuckDB relation —
    DuckDB pushes down filters and projections so the full dataset need not
    fit in memory.  All other inputs (TSV files, stdin, parquet-pipe) are
    read into a pandas DataFrame first and then registered with DuckDB.
    """
    if not _HAS_DUCKDB:
        raise ImportError("--sql requires the duckdb package: pip install duckdb")

    conn = duckdb.connect()
    filename: str | None = getattr(args, "DATAFILE", None)

    # Named parquet: let DuckDB read it lazily (potentially out-of-core).
    direct_parquet = (
        filename
        and filename not in (None, "-")
        and filename.endswith(".parquet")
        and os.path.isfile(filename)
    )

    if direct_parquet:
        rel = conn.read_parquet(filename)
        conn.register(table, rel)
    elif filename is None and not _uses_table(sql, table):
        # SQL reads its own files (FROM 'f.jsonl'): don't wait for stdin.
        pass
    else:
        # stdin, TSV files, or parquet-pipe: read into pandas first.
        df = io.read(args)
        conn.register(table, df)

    try:
        result = conn.execute(sql)
        description = result.description
        df = result.df()
    except Exception as exc:
        raise ValueError(f"DuckDB SQL failed: {exc}") from exc
    df = _hugeint_to_int64(df, description)
    docformats.mark_json_columns(
        df, [d[0] for d in description or [] if str(d[1]) == "JSON"]
    )
    return df


def _uses_table(sql: str, table: str) -> bool:
    """Return True unless DuckDB's parser says *sql* never names *table*.

    Uses json_serialize_sql, which only parses (duckdb.get_table_names also
    binds, and fails on valid SQL such as SELECT pos.*).  On a parse error
    the input is read as before and the query itself reports the error.
    """
    try:
        tree = json.loads(
            duckdb.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        )
    except Exception:  # an old duckdb without json_serialize_sql
        return True
    if tree.get("error"):
        return True
    names: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "BASE_TABLE":
                names.add(node.get("table_name", "").lower())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(tree)
    return table.lower() in names


def _hugeint_to_int64(df: pd.DataFrame, description) -> pd.DataFrame:
    """Return 128-bit integer result columns as Int64 instead of float64.

    DuckDB's SUM of an integer column is a HUGEINT, which pandas has no type
    for, so .df() gives float64 (19.0).  Below 2**53 that float is exact and
    is converted back; from 2**53 on the float may already be rounded, so it
    stays float.
    """
    for i, (_, type_code, *_) in enumerate(description or []):
        if str(type_code) not in ("HUGEINT", "UHUGEINT"):
            continue
        col = df.iloc[:, i]
        if (col.dropna().abs() < 2**53).all():
            df.isetitem(i, col.astype("Int64"))
    return df


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
PANDAS QUERY MODE  (-q / --query)
----------------------------------
Each expression is applied as a boolean row filter in sequence; multiple
-q flags are ANDed together.  Syntax mirrors Python comparisons on column
names:

  dftk query data.tsv -q "x > 0"
  dftk dataset tips -o | dftk query -q "tip > 5"
  dftk dataset tips -o | dftk query -q "day == 'Sat'" -q "size >= 4"
  dftk dataset tips -o | dftk query -q "total_bill > 40 and tip < 5"
  dftk dataset tips -o | dftk query -q "day.str.startswith('S')"

Backtick-quoting handles column names with spaces or special characters
(none of dftk's bundled datasets have one, but yours might):

  dftk query FILE.tsv -q "`my col` > 0"

This is a thin wrapper around pandas DataFrame.query().


SQL MODE  (--sql)
------------------
A full SQL statement is passed to DuckDB.  The input data is available as
a virtual table named "data" (or the name given by --table).

Simple filtering:

  dftk dataset diamonds -o \\
    | dftk query --sql "SELECT * FROM data WHERE price > 15000"
  dftk dataset diamonds -o \\
    | dftk query --sql "SELECT * FROM data WHERE price > 15000" -o \\
    | dftk stat -c price

Aggregation (replaces dftk pivot for simple cases):

  dftk dataset diamonds -o | dftk query --sql \\
    "SELECT cut, AVG(price) AS mean, STDDEV(price) AS std, COUNT(*) AS n
     FROM data GROUP BY cut"

Window functions:

  dftk dataset diamonds -o | dftk query --sql \\
    "SELECT *, ROW_NUMBER() OVER (PARTITION BY cut ORDER BY price DESC) AS rank
     FROM data"

Differences between rows: LAG() and LEAD() read a column in the previous
or next row of the window, so one row's start can be compared with the
previous row's end. Gaps between segments on each chromosome (negative =
overlap; the first segment of each chromosome gets an empty cell):

  dftk query segments.tsv --sql \\
    'SELECT *, "Begin" - LAG("End") OVER (PARTITION BY Chr ORDER BY "Begin")
       AS gap
     FROM data ORDER BY Chr, "Begin"'

  Begin and End are SQL keywords: put such column names in double quotes,
  and the whole statement in single quotes. No sorting is needed first;
  ORDER BY inside OVER sets the row order. The same on a bundled dataset,
  change since the previous record and years between records per country:

  dftk dataset healthexp -o | dftk query --sql \\
    "SELECT Country, Year, Spending_USD,
            ROUND(Spending_USD - LAG(Spending_USD) OVER w, 1) AS change,
            Year - LAG(Year) OVER w AS years
     FROM data WINDOW w AS (PARTITION BY Country ORDER BY Year)
     ORDER BY Country, Year"

CTEs:

  dftk dataset diamonds -o | dftk query --sql \\
    "WITH ranked AS (
       SELECT *, RANK() OVER (PARTITION BY cut ORDER BY price DESC) AS rank
       FROM data
     )
     SELECT * FROM ranked WHERE rank <= 3"

JSON FILES
----------
DuckDB reads JSON itself: name the file in FROM, in single quotes. A file
of records ([{...}, {...}]) and JSON Lines (one object per line) both
work, and 'logs/*.jsonl' reads several files as one table. When the SQL
doesn't use the data table, dftk reads no input, so nothing is piped in:

  printf '%s\\n' '{"id": 1, "pos": {"chr": "1", "start": 100}, "tags": ["x", "y"]}' \\
                '{"id": 2, "pos": {"chr": "2", "start": 200}, "tags": []}' \\
    > records.jsonl
  dftk query --sql "SELECT * FROM 'records.jsonl'"

Nested objects and lists come out as one text cell each, which later
commands can't compute on. Flatten them in the same query: pos.* makes a
column of each field, len() counts a list, unnest() gives a row per item:

  dftk query --sql \\
    "SELECT * EXCLUDE (pos, tags), pos.*, len(tags) AS ntags
     FROM 'records.jsonl'"
  dftk query --sql "SELECT id, unnest(tags) AS tag FROM 'records.jsonl'"
  dftk query --sql \\
    "SELECT * EXCLUDE (pos, tags), pos.* FROM 'records.jsonl'" -o \\
    | dftk stat -c start

In TSV output a list or object that is left nested is shown as one line
of JSON (["x","y"]). A field that is a string in some records and a list
in others gets DuckDB type JSON; its plain strings are shown without
quotes. To get JSON back, give -o FILE.json (or .jsonl) or --format json;
nested values, and such mixed fields, are written as JSON again, and a key
missing from a record is written as null:

  dftk query --sql "SELECT * FROM 'records.jsonl' WHERE pos.chr = '1'" \\
    --format json
  dftk query --sql "SELECT id, tags FROM 'records.jsonl'" -o ids.jsonl

OUT-OF-CORE LARGE FILES
------------------------
When the input is a named .parquet file, DuckDB queries it natively without
loading the full dataset into memory.  Filter pushdown and column pruning
mean that only the rows and columns needed by the query are read.  Use a
named checkpoint file (-o FILE.parquet, not a bare -o) so the file persists
for the query below it — here on the real 54k-row diamonds table:

  dftk dataset diamonds -o diamonds.parquet
  dftk query diamonds.parquet --sql "SELECT * FROM data WHERE price > 15000"
  dftk query diamonds.parquet --sql \\
    "SELECT cut, AVG(price) AS mean FROM data GROUP BY cut"

For piped input (stdin or parquet-pipe), the data is loaded into pandas
first and then registered with DuckDB — in-memory, same as pandas mode.
The same named-checkpoint approach works after a transform step too:

  dftk dataset diamonds -o | dftk eval -f "price_per_carat = price / carat" \\
    -o diamonds_derived.parquet
  dftk query diamonds_derived.parquet --sql \\
    "SELECT * FROM data WHERE price_per_carat > 5000"

Requires: pip install duckdb
"""


class QueryCommand(BaseCommand):
    name = "query"
    help = "Filter rows with pandas expressions (-q) or full DuckDB SQL (--sql)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("query options")
        g.add_argument(
            "-q",
            "--query",
            action="append",
            default=[],
            metavar="EXPR",
            help=(
                "Pandas boolean expression for row filtering.  "
                "Repeatable; all expressions are ANDed together.  "
                "Cannot be combined with --sql."
            ),
        )
        g.add_argument(
            "--sql",
            default=None,
            metavar="SQL",
            help=(
                "Full DuckDB SQL query.  The input is registered as a virtual "
                "table named 'data' (see --table).  Named .parquet inputs are "
                "queried natively without full load into memory.  "
                "SQL that only reads its own files (FROM 'f.jsonl') "
                "needs no input.  "
                "Cannot be combined with -q.  Requires duckdb."
            ),
        )
        g.add_argument(
            "--table",
            default="data",
            metavar="NAME",
            help=(
                "Name of the virtual table in --sql queries (default: data).  "
                "Change this if your data has a column named 'data'."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.sql is not None and args.query:
            raise ValueError("Cannot combine -q/--query and --sql.")

        if args.sql is not None:
            df = _sql_query(args, args.sql, args.table)
        else:
            df = io.read(args)
            if args.query:
                df = _pandas_query(df, args.query)

        io.printdf(df, args)
