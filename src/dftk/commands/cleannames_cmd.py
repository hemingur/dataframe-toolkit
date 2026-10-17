"""
dftk.commands.cleannames_cmd — dftk cleannames subcommand.

Rename columns to lowercase ASCII identifiers (snake_case), so they can be
used unquoted in query --sql, eval -f and fit formulas.  Typical input is a
spreadsheet export with headers such as "Sample ID", "GC %" or "Hæð (cm)".

Unlike R janitor's clean_names, camelCase is not split ("sampleID" becomes
"sampleid", not "sample_id"): splitting would also turn "HbA1c" into
"hb_a1c" and "pH" into "p_h".  "%" and "#" become "pct" and "n".

The original names are kept in the parquet metadata (key original_names, a
JSON object cleaned → original; see dftk annotate) and, with --mapping, in a
TSV file.
"""

import argparse
import json
import logging
import re
import unicodedata

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import io

_EPILOG = """\
RULES
-----
  Sample ID       ->  sample_id    lowercase; spaces and symbols become _
   Age (years)    ->  age_years    trimmed; one _ per run, none at the ends
  GC %            ->  gc_pct       % becomes pct
  Count #         ->  count_n      # becomes n
  1960            ->  x1960        a leading digit gets an x
  Hæð (µg)        ->  haed_ug      letters become plain ASCII
  sampleID        ->  sampleid     camelCase is not split
  ---             ->  colN         nothing left: col + its position (1-based)
  GC % , gc_pct   ->  gc_pct, gc_pct_2   clashes get _2, _3, ...

Duplicate and empty headers are already renamed when the file is read
(Count #.1, Unnamed: 7), so they come out as count_n_1 and unnamed_7.
Names that are SQL reserved words (group, order, end, ...) are kept, with a
warning: quote them in query --sql ("group").

EXAMPLES
--------
  dftk cleannames data.tsv

Convert a spreadsheet export to parquet with clean names, keeping integer
columns with empty cells as integers:

  dftk cleannames data.tsv --backend duckdb -o data.parquet

The World Bank fertility table has headers such as "Country Name" and one
column per year (1960, 1961, ...); after cleaning they can be used
unquoted in SQL:

  dftk dataset fertility -o \\
      | dftk cleannames ... -o \\
      | dftk query ... --sql "SELECT country_name, x1960, x2000 FROM data LIMIT 3"

Write the old and new names to a TSV file, e.g. for a data dictionary:

  dftk dataset fertility -o \\
      | dftk cleannames ... --mapping names.tsv -o fertility.parquet
"""

# Letters that Unicode does not decompose into ASCII letter + accent
_TRANSLIT = str.maketrans(
    {
        "æ": "ae",
        "ð": "d",
        "đ": "d",
        "þ": "th",
        "ø": "o",
        "œ": "oe",
        "ß": "ss",
        "ł": "l",
        "ı": "i",
        "µ": "u",  # micro sign, as in µg
        "μ": "u",  # Greek mu
    }
)
_SYMBOLS = {"%": " pct ", "#": " n "}


def clean_name(name) -> str:
    """Return *name* as a lowercase ASCII identifier ("" if nothing is left)."""
    s = str(name).strip()
    for symbol, word in _SYMBOLS.items():
        s = s.replace(symbol, word)
    s = s.lower().translate(_TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    if s[:1].isdigit():
        s = "x" + s
    return s


def clean_names(names: list) -> list[str]:
    """Clean every name; empty results become colN, clashes get _2, _3, …"""
    cleaned = [clean_name(n) or f"col{i + 1}" for i, n in enumerate(names)]
    taken = set(cleaned)  # a new name must not clash with any cleaned name
    out: list[str] = []
    for name in cleaned:
        new, k = name, 1
        while new in out or (new != name and new in taken):
            k += 1
            new = f"{name}_{k}"
        out.append(new)
    return out


def _sql_keywords(names: list[str]) -> list[str]:
    """Return the names that DuckDB SQL only accepts quoted."""
    try:
        import duckdb
    except ImportError:
        return []
    rows = duckdb.sql(
        "SELECT keyword_name FROM duckdb_keywords() "
        "WHERE keyword_category IN ('reserved', 'type_function')"
    ).fetchall()
    keywords = {r[0] for r in rows}
    return [n for n in names if n in keywords]


class CleannamesCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "cleannames"

    @property
    def help(self) -> str:
        return "Rename columns to lowercase identifiers (sample_id, gc_pct)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("cleannames options")
        g.add_argument(
            "--mapping",
            metavar="FILE",
            default=None,
            help="Also write the names to FILE as TSV (columns original, cleaned).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        original = [str(c) for c in df.columns]
        cleaned = clean_names(original)
        df.columns = cleaned

        # Carry names from an earlier cleannames run, so the first originals
        # (e.g. spreadsheet headers) are kept when a file is cleaned again
        meta = dict(df.attrs.get("_parquet_meta", {}))
        earlier = json.loads(meta.get("original_names", "{}"))
        renamed = {}
        for old, new in zip(original, cleaned, strict=True):
            first = earlier.get(old, old)
            if first != new:
                renamed[new] = first
        if renamed or earlier:
            meta["original_names"] = json.dumps(renamed, ensure_ascii=False)
            df.attrs["_parquet_meta"] = meta

        if args.mapping is not None:
            pd.DataFrame({"original": original, "cleaned": cleaned}).to_csv(
                args.mapping, sep="\t", index=False
            )

        keywords = _sql_keywords(cleaned)
        if keywords:
            quoted = ", ".join(f'"{k}"' for k in keywords)
            logging.warning(
                f"cleannames: {', '.join(keywords)} are SQL reserved words; "
                f"quote them in query --sql ({quoted})"
            )

        io.printdf(df, args)
