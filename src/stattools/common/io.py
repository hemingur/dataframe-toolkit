"""
stattools.common.io — core I/O module for the dfstat toolkit.

Ported from df/src/df/dfio.py with the following changes:

  • -o/--output replaces --ofstream, with three modes:
      absent           → TSV written to stdout
      -o               → temp parquet in DFSTAT_TMPDIR, path printed to stdout
                         (auto-deleted by the next reader in the pipeline)
      -o FILE.parquet  → named parquet written to FILE, path printed to stdout
                         (NOT auto-deleted — reusable later in the same pipeline)
      -o FILE          → TSV written to FILE, nothing on stdout

  • Parquet-pipe via ``...`` DATAFILE: use ``...`` as the DATAFILE argument to
    read a parquet path from the first line of stdin and load that file.
    Temp files (inside DFSTAT_TMPDIR) are deleted after reading; named files
    are left intact so they can be referenced again downstream.

  • Removed --sqlout and --latex (niche output formats).
  • Removed --experimental (unused).
  • parser_printdf kept as an alias for parser_output for compatibility.
"""

import argparse
import csv
import fnmatch
import io as _io
import logging
import os
import sys
import tempfile

import numpy as np
import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None

try:
    import polars as pl
except ImportError:
    pl = None

logging.basicConfig(
    format="%(asctime)s %(module)s %(levelname)s at line %(lineno)d: %(message)s",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
# Parquet-pipe temp directory
# ---------------------------------------------------------------------------

DFSTAT_TMPDIR: str = os.environ.get(
    "DFSTAT_TMPDIR",
    os.path.join(tempfile.gettempdir(), "dfstat"),
)


def _is_pipe_path(path: str) -> bool:
    """Return True if *path* looks like a parquet pipe path arriving on stdin.

    Any existing .parquet file qualifies — both temp files in DFSTAT_TMPDIR
    and named files written by a previous ``-o FILE.parquet`` invocation.
    Only temp files (those inside DFSTAT_TMPDIR) are auto-deleted after reading.
    """
    return path.endswith(".parquet") and os.path.isfile(path)


def _is_temp_pipe_path(path: str) -> bool:
    """Return True if *path* is a dfstat-created temp parquet that should be deleted."""
    return (
        path.endswith(".parquet")
        and os.path.isfile(path)
        and os.path.abspath(os.path.dirname(path)) == os.path.abspath(DFSTAT_TMPDIR)
    )


def _read_parquet_meta(path: str) -> dict[str, str]:
    """Return custom (non-pandas) metadata from a parquet file's schema."""
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    raw = schema.metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items() if k != b"pandas"}


def _write_parquet(
    df: pd.DataFrame, path: str, meta: dict[str, str] | None = None
) -> None:
    """Write *df* to *path* as parquet, embedding custom *meta* annotations.

    Any metadata already carried in ``df.attrs["_parquet_meta"]`` is merged
    with *meta* (explicit *meta* values take precedence).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(df, preserve_index=False)
    # Start from whatever the pandas→arrow conversion embedded
    existing: dict[bytes, bytes] = dict(table.schema.metadata or {})
    # Merge attrs-carried provenance then explicit --meta overrides
    carried: dict[str, str] = df.attrs.get("_parquet_meta", {})
    merged = {
        **existing,
        **{k.encode(): v.encode() for k, v in carried.items()},
        **{k.encode(): v.encode() for k, v in (meta or {}).items()},
    }
    table = table.replace_schema_metadata(merged)
    pq.write_table(table, path)


def _write_pipe_parquet(df: pd.DataFrame) -> str:
    """Write *df* to a temp parquet file in DFSTAT_TMPDIR and return the path."""
    os.makedirs(DFSTAT_TMPDIR, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".parquet", dir=DFSTAT_TMPDIR)
    os.close(fd)
    _write_parquet(df, path)
    return path


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def check_cols(
    df: pd.DataFrame,
    cols: list[str] | None,
    context: str = "",
) -> None:
    """Raise *ValueError* if any column in *cols* is absent from *df*.

    Parameters
    ----------
    df      : DataFrame to validate against.
    cols    : Column names to check.  None or empty → no-op.
    context : Short label shown in the error, e.g. ``"-c/--cols"``.
    """
    if not cols:
        return
    missing = [c for c in cols if c not in df.columns]
    if missing:
        label = f" ({context})" if context else ""
        available = list(df.columns)
        raise ValueError(
            f"Column(s) not found{label}: {missing}.  Available columns: {available}"
        )


def globnames(names: list[str], columns: list[str]) -> list[list[str]]:
    """Shell-style wildcard match of *names* against *columns*."""
    result = []
    for name in names:
        result.append(fnmatch.filter(columns, name))
    return list(dict.fromkeys(result))


# ---------------------------------------------------------------------------
# I/O class
# ---------------------------------------------------------------------------


class io:
    """Namespace for parser factories and read/write helpers."""

    # ------------------------------------------------------------------ #
    # Input                                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parser_read(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        if parser is None:
            parser = argparse.ArgumentParser(description="Read data into a dataframe")

        g = parser.add_argument_group("data input")
        g.add_argument(
            "DATAFILE",
            help=(
                "Input file (tab-separated, or .parquet).  "
                "Omit or use - to read TSV from stdin.  "
                "Use ... to read a parquet path from stdin "
                "(written by a previous -o command)."
            ),
            nargs="?",
            default=None,
        )
        g.add_argument(
            "--backend",
            help="Backend for reading data (default: pandas)",
            choices=["pandas", "duckdb", "polars"],
            default="pandas",
        )
        g.add_argument(
            "--noheader",
            help="File has no header row; columns are named V1, V2, …",
            action="store_true",
        )
        g.add_argument(
            "--nrows",
            help="Maximum number of rows to read",
            type=int,
            default=None,
        )
        g.add_argument(
            "--delimiter",
            help="Column delimiter (default: tab)",
            default=None,
        )
        g.add_argument(
            "--readasobject",
            help=(
                "Read named columns as object dtype.  "
                "Omit column names to apply to all columns."
            ),
            nargs="*",
            default=None,
            metavar="COL",
        )
        g.add_argument(
            "--prequery",
            help=(
                "Pandas query expression applied immediately after reading.  "
                "Repeatable."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        return parser

    @staticmethod
    def read(args: argparse.Namespace) -> pd.DataFrame:
        """Read a dataframe from the source described in *args*."""

        filename: str | None = getattr(args, "DATAFILE", None)
        backend: str = getattr(args, "backend", "pandas")
        sep: str = getattr(args, "delimiter", None) or "\t"
        noheader: bool = getattr(args, "noheader", False)
        header: int | None = None if noheader else 0
        nrows: int | None = getattr(args, "nrows", None)
        readasobject = getattr(args, "readasobject", None)
        prequery: list[str] = getattr(args, "prequery", [])

        dtype = None
        if readasobject is not None:
            dtype = (
                object if len(readasobject) == 0 else {c: object for c in readasobject}
            )

        # -- helpers -------------------------------------------------------

        def _rename_if_noheader(df: pd.DataFrame) -> pd.DataFrame:
            if noheader:
                df.columns = [f"V{i + 1}" for i in range(len(df.columns))]
            return df

        def _csv_from_bytes(data: bytes) -> pd.DataFrame:
            df = pd.read_csv(
                _io.BytesIO(data),
                sep=sep,
                header=header,
                nrows=nrows,
                dtype=dtype,
                engine="pyarrow",
            )
            return _rename_if_noheader(df)

        def _csv_from_file(path: str) -> pd.DataFrame:
            df = pd.read_csv(
                path,
                sep=sep,
                header=header,
                nrows=nrows,
                dtype=dtype,
                engine="pyarrow",
            )
            return _rename_if_noheader(df)

        # -- read ----------------------------------------------------------

        if filename == "...":
            # Parquet-pipe mode: read a path from stdin, load that parquet file.
            pipe_path = sys.stdin.readline().rstrip("\r\n")
            if not pipe_path.endswith(".parquet") or not os.path.isfile(pipe_path):
                raise ValueError(
                    f"'...' expects a .parquet path on stdin, got {pipe_path!r}"
                )
            df = pd.read_parquet(pipe_path)
            df.attrs["_parquet_meta"] = _read_parquet_meta(pipe_path)
            if _is_temp_pipe_path(pipe_path):
                try:
                    os.unlink(pipe_path)
                except OSError as exc:
                    logging.warning(f"Could not remove pipe file {pipe_path!r}: {exc}")

        elif filename is not None and filename != "-":
            # Named file
            if filename.endswith(".parquet"):
                df = pd.read_parquet(filename)
                df.attrs["_parquet_meta"] = _read_parquet_meta(filename)
            elif backend == "duckdb" and duckdb is not None:
                try:
                    df = duckdb.query(
                        "SELECT * FROM read_csv_auto($1, delim=$2)", [filename, sep]
                    ).to_df()
                except Exception as exc:
                    logging.error(f"DuckDB failed to read {filename!r}: {exc}")
                    raise
            elif backend == "polars" and pl is not None:
                try:
                    df = pl.read_csv(filename, separator=sep).to_pandas()
                except Exception as exc:
                    logging.error(f"Polars failed to read {filename!r}: {exc}")
                    raise
            else:
                df = _csv_from_file(filename)

        else:
            # stdin — read TSV directly
            df = _csv_from_bytes(sys.stdin.buffer.read())

        # -- prequery ------------------------------------------------------

        for query in prequery:
            try:
                df = df.query(query)
            except Exception as exc:
                logging.error(f"--prequery failed ({query!r}): {exc}")
                raise

        return df

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def parser_output(
        parser: argparse.ArgumentParser | None = None,
    ) -> argparse.ArgumentParser:
        """Add output options to *parser*.

        The key option is ``-o / --output`` with three modes:

        * absent        → TSV to stdout
        * ``-o``        → write temp parquet, print path to stdout  (pipe mode)
        * ``-o FILE``   → write to FILE (.parquet if extension matches, else TSV)
        """
        if parser is None:
            parser = argparse.ArgumentParser()

        g = parser.add_argument_group("data output")

        g.add_argument(
            "-o",
            "--output",
            nargs="?",
            const="",  # -o with no argument → pipe mode (empty string sentinel)
            default=None,  # absent → TSV to stdout
            metavar="FILE",
            help=(
                "Output destination.  "
                "Omit for TSV on stdout.  "
                "Use -o alone to write a temp parquet to DFSTAT_TMPDIR and print its "
                "path to stdout (auto-deleted by the next reader).  "
                "Use -o FILE.parquet to write a named parquet and print its path to "
                "stdout (not auto-deleted — reusable later in the same pipeline).  "
                "Use -o FILE (no .parquet) to write TSV to FILE with no stdout output."
            ),
        )
        g.add_argument(
            "--digits",
            help="Significant floating-point digits in TSV output",
            type=int,
            default=None,
        )
        g.add_argument(
            "--drop",
            help="Columns to drop before output.  Glob patterns supported.",
            nargs="*",
            default=None,
            metavar="COL",
        )
        g.add_argument(
            "--move",
            help="Rename columns (OLD:NEW).  Applied before --select.",
            nargs="*",
            default=None,
            metavar="OLD:NEW",
        )
        g.add_argument(
            "--select",
            help=(
                "Select and reorder output columns.  Glob patterns supported.  "
                "Append + to include all remaining columns after the listed ones.  "
                "A single FIRST:LAST argument selects a positional range (1-based)."
            ),
            nargs="*",
            default=None,
            metavar="COL",
        )
        g.add_argument(
            "--cast",
            help="Cast column to dtype (COL:TYPE).  Glob patterns supported.",
            nargs="*",
            default=None,
            metavar="COL:TYPE",
        )
        g.add_argument(
            "--round",
            help=(
                "Round columns to N decimal places (COL:N).  "
                "A bare integer N rounds all float columns."
            ),
            nargs="*",
            default=None,
            metavar="COL:N",
        )
        g.add_argument(
            "--removeheader",
            help="Suppress the header row in TSV output.",
            action="store_true",
        )
        g.add_argument(
            "--deduplicate",
            help="Remove duplicate rows, optionally limited to named columns.",
            nargs="*",
            default=None,
            metavar="COL",
        )
        g.add_argument(
            "--postquery",
            help="Pandas query expression applied after processing.  Repeatable.",
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "--meta",
            help=(
                "Embed provenance metadata in parquet output (KEY=VALUE).  "
                "Repeatable.  Ignored for TSV output."
            ),
            default=[],
            action="append",
            metavar="KEY=VALUE",
        )
        g.add_argument(
            "--errortag",
            help="Tag included in error log messages (useful when pipelining).",
            default="-",
        )

        sort_group = g.add_mutually_exclusive_group()
        sort_group.add_argument(
            "--sortasc",
            help="Sort ascending by named columns.",
            nargs="*",
            default=None,
            metavar="COL",
        )
        sort_group.add_argument(
            "--sortdesc",
            help="Sort descending by named columns.",
            nargs="*",
            default=None,
            metavar="COL",
        )
        sort_group.add_argument(
            "--sort",
            help="Sort by multiple columns with mixed directions (COL:a or COL:d).",
            nargs="*",
            default=None,
            metavar="COL:DIR",
        )

        na_group = g.add_mutually_exclusive_group()
        na_group.add_argument(
            "--na_rep",
            help=(
                "Replace NA values in output.  "
                "A single value applies globally; COL:VAL targets a specific column."
            ),
            nargs="*",
            default=None,
            metavar="VAL",
        )
        na_group.add_argument(
            "--dropna",
            help="Drop rows containing any NA value before output.",
            action="store_true",
        )

        return parser

    # Alias so existing code that calls io.parser_printdf() keeps working.
    parser_printdf = parser_output

    @staticmethod
    def printdf(df: pd.DataFrame, args: argparse.Namespace) -> None:
        """Apply output transformations from *args* and write *df*."""

        pd.options.mode.chained_assignment = None

        # ---------------------------------------------------------------- #
        # Pre-output transformations (order matches original dfio.py)       #
        # ---------------------------------------------------------------- #

        if getattr(args, "dropna", False):
            df = df.dropna(axis=0, how="any")

        # Rename columns
        for item in getattr(args, "move", None) or []:
            try:
                old, new = item.split(":")
                df = df.rename(columns={old: new})
            except Exception:
                logging.warning(f"--move: failed to rename from {item!r}")

        # Fill NA
        na_rep = getattr(args, "na_rep", None)
        if na_rep is not None:
            if len(na_rep) == 1 and ":" not in na_rep[0]:
                try:
                    val = np.float64(na_rep[0])
                except Exception:
                    val = na_rep[0]
                df = df.fillna(value=val)
            else:
                na_dict: dict = {}
                for item in na_rep:
                    col, raw = item.split(":")
                    try:
                        raw = np.float64(raw)
                    except Exception:
                        pass
                    na_dict[col] = raw
                df = df.fillna(value=na_dict)

        # Post-query filter
        for query in getattr(args, "postquery", []):
            try:
                df = df.query(query)
            except Exception as exc:
                logging.error(f"--postquery failed ({query!r}): {exc}")
                raise

        # Cast columns
        _DTYPES = ["object", "int64", "float64", "bool", "datetime64", "category"]
        for item in getattr(args, "cast", None) or []:
            try:
                col, coltype = item.split(":")
                coltype = fnmatch.filter(_DTYPES, f"{coltype}*")[0]
                np_type = getattr(np, coltype)
                for c in fnmatch.filter(df.columns, col):
                    try:
                        df[c] = df[c].astype(np_type)
                    except Exception:
                        df[c] = df[c].astype(float).astype(np_type)
            except Exception:
                logging.warning(f"--cast: failed for {item!r}")

        # Sort
        sortcols: list[str] = []
        sortdirs: list[bool] = []
        if getattr(args, "sortasc", None) is not None:
            sortcols += args.sortasc
            sortdirs += [True] * len(args.sortasc)
        if getattr(args, "sortdesc", None) is not None:
            sortcols += args.sortdesc
            sortdirs += [False] * len(args.sortdesc)
        for item in getattr(args, "sort", None) or []:
            col, direction = item.split(":")
            sortcols.append(col)
            sortdirs.append(direction.lower() != "d")
        if sortcols:
            try:
                df = df.sort_values(by=sortcols, ascending=sortdirs)
            except Exception as exc:
                logging.warning(f"--sort failed: {exc}")

        # Drop columns
        for pattern in getattr(args, "drop", None) or []:
            for col in fnmatch.filter(df.columns, pattern):
                try:
                    df = df.drop(columns=[col])
                except KeyError:
                    pass

        # Select / reorder columns.
        # IMPORTANT: this block intentionally runs against the *output* DataFrame
        # that was passed to printdf (i.e. after the command has generated any new
        # columns such as "mean" from stat or "y" from eval).  Never call
        # check_cols() on --select / --move arguments against the
        # *input* DataFrame inside a command, because those columns may not exist
        # there yet.
        select = getattr(args, "select", None)
        if select is not None:
            if len(select) == 1:
                try:
                    first, last = select[0].split(":")
                    select = list(df.columns[int(first) - 1 : int(last)])
                except Exception:
                    pass
            try:
                selcols: list[str] = []
                _GLOB_CHARS = frozenset("*?[")
                for item in select:
                    if item == "+":
                        continue
                    matched = fnmatch.filter(df.columns, item)
                    # Warn when a literal name (no glob chars) matches nothing
                    if not matched and not (_GLOB_CHARS & set(item)):
                        logging.warning(
                            f"--select: column {item!r} not found in DataFrame "
                            f"(available: {list(df.columns)})"
                        )
                    selcols += matched
                if select[-1] == "+":
                    selcols += [c for c in df.columns if c not in selcols]
                df = df[selcols].reset_index(drop=True)
            except Exception as exc:
                logging.error(f"--select failed: {exc}")

        # Round decimal places
        for item in getattr(args, "round", None) or []:
            try:
                col, n = item.split(":")
                for c in fnmatch.filter(df.columns, col):
                    df[c] = df[c].round(int(n))
            except Exception:
                try:
                    n = int(item)
                    for c in df.columns:
                        if df[c].dtype == np.float64:
                            df[c] = df[c].round(n)
                except Exception:
                    pass

        # Deduplicate
        dedup = getattr(args, "deduplicate", None)
        if dedup is not None:
            df = df.drop_duplicates(subset=dedup if dedup else None)

        # ---------------------------------------------------------------- #
        # Output routing                                                    #
        # ---------------------------------------------------------------- #

        output: str | None = getattr(args, "output", None)
        digits: int | None = getattr(args, "digits", None)
        float_format = f"%.{digits}g" if digits is not None else None

        # Parse --meta KEY=VALUE pairs
        meta: dict[str, str] = {}
        for item in getattr(args, "meta", []):
            try:
                k, v = item.split("=", 1)
                meta[k.strip()] = v.strip()
            except ValueError:
                logging.warning(f"--meta: expected KEY=VALUE, got {item!r}")

        # Header: suppressed if the input had no header, or --removeheader set
        header = not getattr(args, "noheader", False)
        if header and getattr(args, "removeheader", False):
            header = False

        if output == "":
            # Pipe mode — write temp parquet, print path
            os.makedirs(DFSTAT_TMPDIR, exist_ok=True)
            fd, path = tempfile.mkstemp(suffix=".parquet", dir=DFSTAT_TMPDIR)
            os.close(fd)
            _write_parquet(df, path, meta=meta or None)
            print(path)

        elif output is not None and output.endswith(".parquet"):
            # Named parquet — write file and print path so the pipe continues.
            # The file is NOT auto-deleted by downstream readers (only DFSTAT_TMPDIR
            # temp files are deleted), so it can be referenced later in a pipeline.
            _write_parquet(df, output, meta=meta or None)
            print(output)

        else:
            # TSV — stdout or named file
            dest = output if output is not None else sys.stdout
            try:
                df.to_csv(
                    dest,
                    sep="\t",
                    index=False,
                    float_format=float_format,
                    header=header,
                    quoting=csv.QUOTE_NONE,
                    doublequote=False,
                )
            except (OSError, BrokenPipeError):
                sys.stderr.close()
