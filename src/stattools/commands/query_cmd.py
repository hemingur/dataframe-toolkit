"""
stattools.commands.query_cmd — dfstat query subcommand.

Filter (and optionally reshape) a dataframe using either pandas query
expressions or full DuckDB SQL.

Two modes
---------
pandas mode  (-q / --query, repeatable)
    Applies pandas DataFrame.query() expressions in sequence.  Each
    expression is a boolean row filter; all expressions are ANDed together.

      dfstat query data.tsv -q "x > 0" -q "group == 'A'"

sql mode  (--sql)
    Passes a full SQL statement to DuckDB.  The input data is available as
    a virtual table named "data" (override with --table).  Any SQL that
    DuckDB supports is valid: SELECT, aggregation, window functions, CTEs.

      dfstat query data.tsv --sql "SELECT grp, AVG(x) AS mean FROM data GROUP BY grp"

    When the input is a named .parquet file, DuckDB queries it natively
    without loading the full file into memory — suitable for large datasets.

    Requires: pip install duckdb
"""

import argparse
import logging
import os

import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io

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
    else:
        # stdin, TSV files, or parquet-pipe: read into pandas first.
        df = io.read(args)
        conn.register(table, df)

    try:
        return conn.execute(sql).df()
    except Exception as exc:
        raise ValueError(f"DuckDB SQL failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

_EPILOG = """\
PANDAS QUERY MODE  (-q / --query)
----------------------------------
Each expression is applied as a boolean row filter in sequence; multiple
-q flags are ANDed together.  Syntax mirrors Python comparisons on column
names:

  dfstat query data.tsv -q "x > 0"
  dfstat query data.tsv -q "group == 'A'" -q "score >= 10"
  dfstat query data.tsv -q "x > 0 and y < 100"
  dfstat query data.tsv -q "label.str.startswith('foo')"

Backtick-quoting handles column names with spaces or special characters:

  dfstat query data.tsv -q "`my col` > 0"

This is a thin wrapper around pandas DataFrame.query().


SQL MODE  (--sql)
------------------
A full SQL statement is passed to DuckDB.  The input data is available as
a virtual table named "data" (or the name given by --table).

Simple filtering:

  dfstat query data.tsv --sql "SELECT * FROM data WHERE x > 0"
  cat data.tsv | dfstat query --sql "SELECT * FROM data WHERE x > 0" -o

Aggregation (replaces dfstat pivot for simple cases):

  dfstat query data.tsv --sql \\
    "SELECT group, AVG(x) AS mean, STDDEV(x) AS std, COUNT(*) AS n
     FROM data GROUP BY group"

Window functions:

  dfstat query data.tsv --sql \\
    "SELECT *, ROW_NUMBER() OVER (PARTITION BY group ORDER BY x DESC) AS rank
     FROM data"

CTEs:

  dfstat query data.tsv --sql \\
    "WITH ranked AS (
       SELECT *, RANK() OVER (PARTITION BY group ORDER BY score DESC) AS rank
       FROM data
     )
     SELECT * FROM ranked WHERE rank <= 3"

OUT-OF-CORE LARGE FILES
------------------------
When the input is a named .parquet file, DuckDB queries it natively without
loading the full dataset into memory.  Filter pushdown and column pruning
mean that only the rows and columns needed by the query are read:

  dfstat query huge.parquet --sql "SELECT * FROM data WHERE x > 0"
  dfstat query huge.parquet --sql \\
    "SELECT group, AVG(x) AS mean FROM data GROUP BY group"

For piped input (stdin or parquet-pipe), the data is loaded into pandas
first and then registered with DuckDB — in-memory, same as pandas mode.
Use a named checkpoint file (-o huge.parquet) when you need out-of-core
queries on a large intermediate result:

  dfstat eval "y = x * 2" huge.tsv -o huge.parquet
  dfstat query huge.parquet --sql "SELECT * FROM data WHERE y > 1000"

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
