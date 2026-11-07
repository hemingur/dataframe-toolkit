"""
dftk.common.docformats — JSON, Markdown and LaTeX output for io.printdf.

Nested values (lists, structs, maps) come from DuckDB as numpy arrays and
dicts.  TSV, Markdown and LaTeX show them as one line of JSON text; JSON
output writes them as JSON.

A DuckDB column of type JSON (from a JSON file whose field is a string in one
record and a list in another) reaches pandas as JSON text, e.g.
'"chrX:1-2"' and '["chr3:1-2","chr3:2-3"]'.  query_cmd lists such columns in
the parquet metadata key JSON_COLUMNS_KEY, so they survive -o pipes; the text
view drops the quotes around a plain string and JSON output parses them.
"""

import contextlib
import decimal
import json
import math

import numpy as np
import pandas as pd

JSON_COLUMNS_KEY = "dftk_json_columns"

# Output format by file extension; --format names the same formats
EXTENSIONS = {".json": "json", ".jsonl": "jsonl", ".md": "md", ".tex": "tex"}
FORMATS = ("tsv", "json", "jsonl", "md", "tex")


def json_columns(df: pd.DataFrame) -> list[str]:
    """Columns of *df* that hold DuckDB JSON text (see module docstring)."""
    names = df.attrs.get("_parquet_meta", {}).get(JSON_COLUMNS_KEY, "")
    return [c for c in names.split(",") if c in df.columns]


def mark_json_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Record *columns* as JSON text in df.attrs (carried into parquet)."""
    meta = dict(df.attrs.get("_parquet_meta", {}))
    if columns:
        meta[JSON_COLUMNS_KEY] = ",".join(columns)
    else:
        meta.pop(JSON_COLUMNS_KEY, None)
    df.attrs["_parquet_meta"] = meta


def rename_json_columns(df: pd.DataFrame, mapping: dict[str, str]) -> None:
    """Follow a --move rename in the JSON column list."""
    names = df.attrs.get("_parquet_meta", {}).get(JSON_COLUMNS_KEY)
    if names:
        mark_json_columns(df, [mapping.get(c, c) for c in names.split(",")])


def _plain(value):
    """json.dumps fallback for numpy, Decimal and other non-JSON values."""
    if isinstance(value, np.ndarray):  # MaskedArray.tolist() gives None
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, decimal.Decimal):
        return float(value)
    return str(value)


def _is_nested(value) -> bool:
    return isinstance(value, (np.ndarray, list, tuple, dict))


def _dumps(value) -> str:
    text = json.dumps(value, default=_plain, ensure_ascii=False, separators=(",", ":"))
    if "NaN" in text or "Infinity" in text:
        # NaN and Infinity aren't JSON: write them as null
        parsed = json.loads(text, parse_constant=lambda _: None)
        text = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    return text


def nested_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with nested values and JSON-column text as one-line text."""
    jcols = set(json_columns(df))
    out = None
    # By position: column names may repeat
    for i, col in enumerate(df.columns):
        series = df.iloc[:, i]
        dtype = series.dtype
        if not (
            pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        ):
            continue
        if col in jcols:
            new = series.map(_json_text_cell)
        elif series.map(_is_nested).any():
            new = series.map(lambda v: _dumps(v) if _is_nested(v) else v)
        else:
            continue
        if out is None:
            out = df.copy()
        out.isetitem(i, new)
    return df if out is None else out


def _json_text_cell(value):
    """'"chrX:1-2"' → 'chrX:1-2'; a list or object stays JSON text."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
    except ValueError:
        return value
    return parsed if isinstance(parsed, str) else value


def _json_value(value, digits: int | None, is_json_text: bool):
    """A cell as a JSON-ready Python value."""
    if is_json_text and isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    if _is_nested(value):
        return json.loads(_dumps(value))
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if digits is not None:
            return float(f"{value:.{digits}g}")
        return value
    if isinstance(value, (int, bool, str)):
        return value
    return _plain(value)


def write_json(df: pd.DataFrame, dest, lines: bool, digits: int | None) -> None:
    """Write *df* as an array of records, or one record per line (*lines*).

    A missing value is written as null; the key is kept.
    """
    repeated = df.columns[df.columns.duplicated()].unique().tolist()
    if repeated:
        raise ValueError(
            f"JSON output needs unique column names; repeated: {repeated} "
            "(rename one with --move OLD:NEW)"
        )
    jcols = set(json_columns(df))
    cols = list(df.columns)
    records = (
        {
            col: _json_value(value, digits, col in jcols)
            for col, value in zip(cols, row, strict=True)
        }
        for row in df.itertuples(index=False, name=None)
    )
    with _open(dest) as fh:
        if lines:
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            items = [json.dumps(r, ensure_ascii=False) for r in records]
            fh.write("[\n" + ",\n".join("  " + s for s in items) + "\n]\n")


def _cell_text(value, float_format: str | None) -> str:
    if value is None or value is pd.NA or value is pd.NaT:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return ""
        return float_format % value if float_format else str(float(value))
    return str(value)


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _table(df: pd.DataFrame, float_format: str | None, escape):
    """Escaped header and rows of *df* (nested values as JSON text)."""
    df = nested_as_text(df)
    header = [escape(str(c)) for c in df.columns]
    rows = [
        [escape(_cell_text(v, float_format)) for v in row]
        for row in df.itertuples(index=False, name=None)
    ]
    numeric = [_is_numeric(df.iloc[:, i]) for i in range(df.shape[1])]
    return header, rows, numeric


def _escape_markdown(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def write_markdown(
    df: pd.DataFrame, dest, float_format: str | None, header: bool
) -> None:
    """Write *df* as a pipe table; numeric columns are right-aligned.

    A Markdown table always has a header row; with --removeheader it is empty.
    """
    names, rows, numeric = _table(df, float_format, _escape_markdown)
    if not header:
        names = [""] * len(names)
    rule = ["---:" if num else "---" for num in numeric]
    with _open(dest) as fh:
        for cells in [names, rule, *rows]:
            fh.write("| " + " | ".join(cells) + " |\n")


_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    # A row starting with [ after \\ would be read as \\[length]
    "[": "{[}",
    "]": "{]}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    # The default OT1 font prints these as an em dash and as inverted ! and ?
    "|": r"\textbar{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "\n": " ",
}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in text)


def write_latex(df: pd.DataFrame, dest, float_format: str | None, header: bool) -> None:
    """Write *df* as a bare booktabs tabular (needs \\usepackage{booktabs}).

    Numeric columns are right-aligned (r), others left (l).  Wrap it in your
    own table environment with \\input{file.tex}.  A " (in JSON lists, say)
    and ~ ^ print as intended with \\usepackage[T1]{fontenc}; the default
    OT1 font has no straight double quote.
    """
    names, rows, numeric = _table(df, float_format, _escape_latex)
    spec = "".join("r" if num else "l" for num in numeric)
    with _open(dest) as fh:
        fh.write(f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n")
        if header:
            fh.write(" & ".join(names) + " \\\\\n\\midrule\n")
        for cells in rows:
            fh.write(" & ".join(cells) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


def _open(dest):
    """Open *dest* for writing if it is a path; use it as is if a stream."""
    if isinstance(dest, str):
        return open(dest, "w", encoding="utf-8", newline="")
    return contextlib.nullcontext(dest)
