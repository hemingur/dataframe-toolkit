"""
dftk.commands.append_cmd — dftk append subcommand.

Add one row to a table, with values from -c COL=VALUE; columns not given are
empty (or --fill).  Without input it makes a one-row table.  --inplace adds
the row to a file, creating it if missing: a status log that parallel jobs
can write to, since the file's directory is locked while it is written.

In place, a TSV row whose columns are all in the header, and any JSON Lines
row, is added as one line at the end of the file.  A new column, or a
parquet file, means the file is read and rewritten (to a temp file that then
replaces it, so an interrupted run leaves the old file).
"""

import argparse
import contextlib
import csv
import io as _io
import json
import logging
import os
import re
import shutil
import sys
import tempfile

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common import docformats
from dftk.common.io import _read_parquet_meta, _write_parquet, _write_tsv, io

logger = logging.getLogger(__name__)

_INT = re.compile(r"[+-]?\d+")
_FLOAT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

# Options that still apply with --inplace; any other read or output option
# given with it is an error (--prequery would drop rows from the file)
_INPLACE_OPTIONS = {"DATAFILE", "backend", "digits"}


def parse_value(text: str):
    """A -c value: number, true/false, "quoted text", or text; '' is missing."""
    if text == "":
        return None
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    if _INT.fullmatch(text):
        return int(text)
    if _FLOAT.fullmatch(text):
        return float(text)
    return text


def _row(args: argparse.Namespace) -> tuple[dict, dict]:
    """The new row as {column: value}, and the text each value was given as."""
    values: dict = {}
    texts: dict = {}
    for item in args.constant:
        col, sep, text = item.partition("=")
        col, text = col.strip(), text.strip()
        if not sep or not col:
            raise ValueError(f"-c expects COL=VALUE, got {item!r}")
        values[col] = parse_value(text)
        texts[col] = text
    if args.timestamp is not None:
        # ISO 8601 text, no space: one field for awk and cut; read back as
        # a datetime by dftk, DuckDB and pandas
        now = pd.Timestamp.now().floor("s").isoformat()
        values[args.timestamp] = now
        texts[args.timestamp] = now
    if not values:
        raise ValueError("nothing to append: give -c COL=VALUE or --timestamp")
    return values, texts


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _add_value(old: pd.Series, value, text: str) -> pd.Series:
    """*old* with *value* added at the end, keeping old's type where it can."""
    old = old.reset_index(drop=True)
    if not old.notna().any():
        # No type to keep: infer one (Int64, Float64, boolean, string, ...)
        values = [None] * len(old) + [value]
        try:
            return pd.Series(pd.array(values), name=old.name)
        except (TypeError, ValueError):
            return pd.Series(values, dtype=object, name=old.name)
    if value is not None:
        if _is_text(old) and not isinstance(value, str):
            value = text  # an id column of "007"s stays text
        elif isinstance(value, str) and not _is_text(old):
            logger.warning(
                f"append: column {old.name!r} has no text values; {text!r} "
                "makes it a text column"
            )
            old = old.map(lambda v: None if pd.isna(v) else str(v)).astype(object)
    # Types that can hold a missing value, for the new cell
    if pd.api.types.is_bool_dtype(old) and old.dtype == bool:
        old = old.astype("boolean")
    elif pd.api.types.is_integer_dtype(old) and old.dtype.kind in "iu":
        old = old.astype("Int64")
    out = old.reindex(range(len(old) + 1))
    if value is not None:
        try:
            out.iloc[-1] = value
        except (TypeError, ValueError):
            out = out.astype(object)
            out.iloc[-1] = value
    return out


def _combine(df: pd.DataFrame, values: dict, texts: dict, fill) -> pd.DataFrame:
    """*df* with the new row at the end; new columns go last, empty above."""
    fill_text = "" if fill is None else str(fill)
    columns = list(df.columns) + [c for c in values if c not in df.columns]
    out = {}
    for col in columns:
        old = df[col] if col in df.columns else pd.Series([fill] * len(df), name=col)
        value = values.get(col, fill)
        out[col] = _add_value(old, value, texts.get(col, fill_text))
    result = pd.DataFrame(out, columns=columns)
    result.attrs = dict(df.attrs)
    return result


def _check_strict(names, values: dict, path: str) -> None:
    new = [c for c in values if c not in names]
    if new:
        raise ValueError(f"--strict: {new} not in {path} (columns: {list(names)})")


# ---------------------------------------------------------------------------
# --inplace
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _locked(path: str):
    """Hold an exclusive lock on *path*'s directory (no lock file left behind).

    The directory, not the file, is locked: a rewrite replaces the file, and a
    lock on the old one would not stop a second writer.
    """
    try:
        import fcntl
    except ImportError:  # Windows: no locking
        yield
        return
    fd = os.open(os.path.dirname(os.path.abspath(path)), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # releases the lock


def _replace(path: str, write) -> None:
    """Write a new *path* with write(tmp_path), then swap it in."""
    folder = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=folder)
    os.close(fd)
    try:
        write(tmp)
        if os.path.exists(path):
            shutil.copymode(path, tmp)
        else:  # mkstemp makes it private; use the usual mode
            umask = os.umask(0)
            os.umask(umask)
            os.chmod(tmp, 0o666 & ~umask)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def _tsv_text(df: pd.DataFrame, digits: int | None, header: bool) -> str:
    """*df* as io.printdf writes TSV."""
    buf = _io.StringIO()
    _write_tsv(
        docformats.nested_as_text(df),
        buf,
        sep="\t",
        index=False,
        float_format=f"%.{digits}g" if digits is not None else None,
        header=header,
        quoting=csv.QUOTE_NONE,
        doublequote=False,
    )
    return buf.getvalue()


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _ends_with_newline(path: str) -> bool:
    with open(path, "rb") as fh:
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def _append_text(path: str, text: str) -> None:
    """Add *text* (whole lines) at the end of *path*, after a newline."""
    if os.path.getsize(path) and not _ends_with_newline(path):
        text = "\n" + text
    with open(path, "a", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _inplace_tsv(path: str, values, texts, args) -> None:
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, encoding="utf-8", newline="") as fh:
            names = fh.readline().rstrip("\r\n").split("\t")
        if args.strict:
            _check_strict(names, values, path)
        if all(c in names for c in values) and len(set(names)) == len(names):
            # The usual case: one more line at the end
            row = {c: values.get(c, args.fill) for c in names}
            row_df = _combine(pd.DataFrame(columns=names), row, texts, args.fill)
            _append_text(path, _tsv_text(row_df, args.digits, header=False))
            return
        # As text: the rows above are written back exactly as they were
        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_values=[""],
            quoting=csv.QUOTE_NONE,
        )
    else:
        df = pd.DataFrame()
    out = _combine(df, values, texts, args.fill)
    text = _tsv_text(out, args.digits, header=True)
    _replace(path, lambda tmp: _write_text(tmp, text))


def _inplace_parquet(path: str, values, texts, args) -> None:
    if args.timestamp is not None:  # parquet has a timestamp type
        values = {**values, args.timestamp: pd.Timestamp(values[args.timestamp])}
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df.attrs["_parquet_meta"] = _read_parquet_meta(path)
        if args.strict:
            _check_strict(df.columns, values, path)
    else:
        df = pd.DataFrame()
    out = _combine(df, values, texts, args.fill)
    _replace(path, lambda tmp: _write_parquet(out, tmp))


def _inplace_jsonl(path: str, values, texts, args) -> None:
    if args.strict and os.path.exists(path):
        names: set = set()
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    names.update(json.loads(line))
        _check_strict(names, values, path)
    # Records need not share keys: no fill, nothing to rewrite
    row_df = _combine(pd.DataFrame(), values, texts, None)
    buf = _io.StringIO()
    docformats.write_json(row_df, buf, lines=True, digits=args.digits)
    if os.path.exists(path):
        _append_text(path, buf.getvalue())
    else:
        _write_text(path, buf.getvalue())


def _inplace(args: argparse.Namespace, values: dict, texts: dict) -> None:
    path = args.DATAFILE
    if path in (None, "-", "..."):
        raise ValueError("--inplace needs a FILE to add the row to")
    others = [
        f"--{dest}"
        for dest, default in args.io_defaults.items()
        if dest not in _INPLACE_OPTIONS and getattr(args, dest, default) != default
    ]
    if others:
        raise ValueError(f"--inplace can't be combined with {', '.join(others)}")
    if args.strict and not os.path.exists(path):
        raise ValueError(f"--strict: {path} does not exist")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json", ".md", ".tex"):
        raise ValueError(
            f"--inplace works on TSV, .jsonl and .parquet files, not {ext} "
            "(a JSON log is best kept as .jsonl)"
        )
    with _locked(path):
        if ext == ".parquet":
            _inplace_parquet(path, values, texts, args)
        elif ext == ".jsonl":
            _inplace_jsonl(path, values, texts, args)
        else:
            _inplace_tsv(path, values, texts, args)


def _has_stdin_data() -> bool:
    """True if stdin is a pipe or file with data (waits for it, or for EOF)."""
    try:
        if sys.stdin is None or sys.stdin.isatty():
            return False
        return bool(sys.stdin.buffer.peek(1))
    except (AttributeError, OSError, ValueError):  # no real stdin (pytest)
        return False


_EPILOG = """\
VALUES
------
-c COL=VALUE, repeatable.  A value is a number (42, 1.5, 1e-3), true or
false, or text; quote it to keep it text ('"007"'), and leave it empty for a
missing value (note=).  Without input, append makes a one-row table:

  dftk append -c sample=S12 -c reads=48213 -c status=ok
  dftk append -c sample=S12 -c reads=48213 -c ok=true --format json

ADDING TO A TABLE
-----------------
Columns not given are left empty (--fill VALUE fills them), and a new column
goes at the end, empty in the rows above.  A value keeps its column's type
where it can ('007' stays text in a text column):

  dftk append data.tsv -c id=x99 -c score=0.5
  dftk dataset tips -o | dftk append -c total_bill=20 -c tip=4 -c day=Sun \\
    | tail -n 3

STATUS LOGS (--inplace)
-----------------------
--inplace adds the row to the file itself, creating it if missing.  Parallel
jobs can write to the same log: the file's directory is locked while each row
is written.  --timestamp [COL] adds the local time as ISO 8601 text
(2026-11-09T19:13:57; a timestamp in .parquet), and --strict
refuses columns the file doesn't have, so a typo can't start a new one:

  dftk append runs.tsv --inplace -c step=align -c status=ok --timestamp
  dftk append runs.tsv --inplace -c step=call -c status=ok --timestamp --strict

A .jsonl log needs no matching columns at all: each line is one record.
Read it back with query --sql:

  dftk append runs.jsonl --inplace -c step=align -c reads=48213 --timestamp
  dftk query --sql "SELECT * FROM 'runs.jsonl'"

A row whose columns are all in a TSV header, and any .jsonl row, is written
as one more line; a new column, or a .parquet file, rewrites the file.  For
many rows at once, use concat.
"""


class AppendCommand(BaseCommand):
    name = "append"
    help = "Add one row (-c COL=VALUE); --inplace adds it to a file, e.g. a log."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        first = len(parser._actions)
        self.add_io_arguments(parser)
        parser.set_defaults(
            io_defaults={a.dest: a.default for a in parser._actions[first:]}
        )

        g = parser.add_argument_group("append")
        g.add_argument(
            "-c",
            "--col",
            dest="constant",
            action="append",
            default=[],
            metavar="COL=VALUE",
            help="A value of the new row; repeatable.",
        )
        g.add_argument(
            "--inplace",
            action="store_true",
            help="Add the row to DATAFILE itself (created if missing).",
        )
        g.add_argument(
            "--timestamp",
            nargs="?",
            const="timestamp",
            default=None,
            metavar="COL",
            help="Add the current local time, to the second, as ISO 8601 text "
            "(2026-11-09T19:13:57) (column: timestamp).",
        )
        g.add_argument(
            "--fill",
            default=None,
            metavar="VALUE",
            help="Value for columns the row doesn't give, and for a new "
            "column in the rows above (default: empty).",
        )
        g.add_argument(
            "--strict",
            action="store_true",
            help="Refuse columns the table doesn't have.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        values, texts = _row(args)
        args.fill = None if args.fill is None else parse_value(args.fill)
        if args.inplace:
            _inplace(args, values, texts)
            return
        if args.DATAFILE is not None or _has_stdin_data():
            df = io.read(args)
        else:
            df = pd.DataFrame()
        if args.strict:
            _check_strict(df.columns, values, args.DATAFILE or "the input")
        io.printdf(_combine(df, values, texts, args.fill), args)
