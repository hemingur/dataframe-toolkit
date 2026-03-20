"""
stattools.commands.eval_cmd — dfstat eval subcommand.

Evaluates formulas, constants, and string/row functions against a DataFrame.
Three formula types are supported, each with its own flag:

  -f  pandas eval expressions  (df.eval, falls back to special_function)
  -c  constant column assignments
  -s  string / row functions
"""

import argparse
import fnmatch
import hashlib
import logging
import os
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io
from stattools.common import stats as _stats
from stattools.common import seq as _seq

logger = logging.getLogger(__name__)

# Optional dependency: python-Levenshtein
try:
    import Levenshtein as _lev
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False

_EPILOG = """\
FORMULA EXPRESSIONS (-f)
-------------------------
Each -f argument is first passed to pandas df.eval().  This handles
arithmetic and boolean expressions directly on column names:

  dfstat eval data.tsv -f "z = x + y * 2"
  dfstat eval data.tsv -f "flag = (pval < 0.05) & (effect > 0)"

If df.eval() fails (e.g. for function calls), the expression is retried
as a special function call using the syntax:

  dest = func(col1, col2, ...)

Available special functions:

  Row aggregation (applied across named columns):
    sum  mean  std  min  max  median

  Conditional assignment:
    where(cond_col, true_val, false_val)
      — np.where-style conditional; cond_col must be a boolean column.
        true_val and false_val can each be: a column name, a quoted
        string literal ("foo"), or a numeric literal.
      Example:
        dfstat eval data.tsv \
          -f "big = value > 100" \
          -f "label = where(big, \"large\", \"small\")"

  Bitwise:
    applymask(col, mask_col)     — col & mask_col
    overlap(lo1, hi1, lo2, hi2)  — length of interval overlap

  Column aggregation (glob pattern):
    colsum(col*)     — sum of all columns matching the glob pattern

  NumPy scalars:
    sign(col)

  Path functions (applied element-wise):
    basename  dirname  exists  getsize  realpath

  Statistical functions (see common/stats.py):
    binom_test(k, n, p)
    fisher_test(a, b, c, d)      — p-value
    fisher_OR(a, b, c, d)        — odds ratio
    boschloo_test(a, b, c, d)    — p-value
    boschloo_OR(a, b, c, d)      — odds ratio
    pval2se(effect, pval)
    t2pval(t, df)
    chi2_to_neglogp(chi2, df)
    neglogp_to_chi2(neglogp, df)
    pval_to_chi2(pval, df)
    generalized_poisson_nll(x, lam, theta)


CONSTANT COLUMNS (-c)
----------------------
Add a column with a fixed value:

  dfstat eval data.tsv -c "batch = 3"
  dfstat eval data.tsv -c "label = control"

The value is coerced to numeric when possible.


STRING / ROW FUNCTIONS (-s)
----------------------------
Apply a string or row-level function using the same dest = func(cols) syntax:

  String operations:
    translate(col, from, to)
    len(col)
    join(col1, col2, ...)        — joined with --joinsep (default: "")
    lower(col)  upper(col)
    zfill(col, width)
    find(col, substr)
    match(col, pattern_col)
    substr(col, start, length)
    substring(col, start_col, length)
    replace(col, old, new)
    leftsplit(col, sep, idx)     — split on first sep; idx=0 left part, 1 right
    rightsplit(col, sep, idx)    — split on last sep
    sort(col)                    — sort characters in string
    ord(col)                     — minimum ord value of characters
    template(...)                — Python str.format_map across all columns

  Path operations:
    basename(col)  stem(col)  realpath(col)  isfile(col)  exists(col)

  Crypto / hash:
    md5(col)

  DNA / sequence (see common/seq.py):
    gccontent(col)
    basecount(col)
    motifcount(pattern, seq_col)
    readoffset(read_pos, ref_pos, cigar, seq, count)

  Edit distance (requires python-Levenshtein):
    levenshtein(col1, col2)
"""

# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------


def _parse_formula(formula: str) -> tuple[str, str, list[str], str]:
    """Parse 'dest = func(col1, col2, ...)' into (dest, func, cols, body).

    Returns the raw argument string as *body* so callers can use it as a
    glob pattern (e.g. for colsum).
    """
    try:
        dest, right = (x.strip() for x in formula.split("=", 1))
        func, right = (x.strip() for x in right.split("(", 1))
        body, _ = (x.strip() for x in right.rsplit(")", 1))
    except Exception:
        raise ValueError(f"Bad formula syntax: {formula!r}  "
                         "Expected: dest = func(col1, col2, ...)")
    cols = [x.strip() for x in body.split(",")]
    return dest, func, cols, body


# ---------------------------------------------------------------------------
# Special functions (used as pandas agg fallback inside -f)
# ---------------------------------------------------------------------------

_STATS_DISPATCH = {
    "binom_test":              _stats.binom_test,
    "fisher_test":             _stats.fisher_test,
    "fisher_OR":               _stats.fisher_OR,
    "boschloo_test":           _stats.boschloo_test,
    "boschloo_OR":             _stats.boschloo_OR,
    "pval2se":                 _stats.pval2se,
    "t2pval":                  _stats.t2pval,
    "chi2_to_neglogp":         _stats.chi2_to_neglogp,
    "neglogp_to_chi2":         _stats.neglogp_to_chi2,
    "pval_to_chi2":            _stats.pval_to_chi2,
    "generalized_poisson_nll": _stats.generalized_poisson_nll,
}

_PATH_FUNCS    = {"basename", "dirname", "exists", "getsize", "realpath"}
_NP_FUNCS      = {"sign"}
_ROW_AGG_FUNCS = {"sum", "mean", "std", "min", "max", "median"}
_ROW_IDX_FUNCS = {"idxmax", "idxmin"}


class _FrameFunc:
    """Callable that applies a named pandas DataFrame method along axis=1.

    Used for idxmax/idxmin, which need the column labels and therefore cannot
    go through the standard ``func(row.to_numpy())`` dispatch path.
    """
    def __init__(self, method: str) -> None:
        self.method = method

    def __call__(self, df_subset):
        return getattr(df_subset, self.method)(axis=1)


def _resolve_where_val(val: str, df: pd.DataFrame):
    """Resolve a where() argument to a scalar, Series, or literal string.

    Resolution order:
      1. Quoted string literal  ("foo" or 'foo')
      2. Column name present in *df*
      3. Integer literal
      4. Float literal
      5. Bare string (returned as-is)
    """
    s = val.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in df.columns:
        return df[s]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


class _WhereFunc:
    """Callable for ``where(cond_col, true_val, false_val)`` using ``np.where``.

    Receives the full DataFrame so that *true_val* and *false_val* can be
    resolved as column references, quoted string literals, or numeric literals.
    """
    def __init__(self, cond_col: str, true_val: str, false_val: str) -> None:
        self.cond_col = cond_col
        self.true_val = true_val
        self.false_val = false_val

    def __call__(self, df: pd.DataFrame):
        cond = df[self.cond_col]
        true_res = _resolve_where_val(self.true_val, df)
        false_res = _resolve_where_val(self.false_val, df)
        return np.where(cond, true_res, false_res)


_VALID_SPECIAL = (
    {"sum", "mean", "std", "min", "max", "median",
     "overlap", "applymask", "colsum", "where"}
    | _ROW_IDX_FUNCS
    | _PATH_FUNCS
    | _NP_FUNCS
    | set(_STATS_DISPATCH)
)


def _special_function(formula: str, df_columns: list[str]):
    """Return (dest, func, cols) for a special-function formula."""
    dest, func_name, cols, body = _parse_formula(formula)

    if func_name not in _VALID_SPECIAL:
        raise ValueError(f"Function {func_name!r} is not available as a "
                         "special function.  See dfstat eval --help.")

    if func_name == "where":
        if len(cols) != 3:
            raise ValueError(
                "where() requires exactly 3 arguments: "
                "where(cond_col, true_val, false_val)"
            )
        return dest, _WhereFunc(cols[0], cols[1], cols[2]), []

    if func_name == "overlap":
        func = partial(lambda x: max(0, min(x[1], x[3]) - max(x[0] - 1, x[2] - 1)))
    elif func_name == "applymask":
        func = partial(lambda x: x[0] & x[1])
    elif func_name == "colsum":
        cols = fnmatch.filter(df_columns, body)
        func = np.sum
    elif func_name in _ROW_AGG_FUNCS:
        func = getattr(np, func_name)
        # Each argument may be a glob pattern: expand to matching column names
        cols = [c for pat in cols for c in (fnmatch.filter(df_columns, pat) or [pat])]
    elif func_name in _ROW_IDX_FUNCS:
        func = _FrameFunc(func_name)
        cols = [c for pat in cols for c in (fnmatch.filter(df_columns, pat) or [pat])]
    elif func_name in _NP_FUNCS:
        func = getattr(np, func_name)
    elif func_name in _PATH_FUNCS:
        func = np.frompyfunc(getattr(os.path, func_name), 1, 1)
    elif func_name in _STATS_DISPATCH:
        func = _STATS_DISPATCH[func_name]
    else:
        func = func_name  # bare string passed to pandas agg

    return dest, func, cols


# ---------------------------------------------------------------------------
# String / row functions (used by -s)
# ---------------------------------------------------------------------------


def _string_function(formula: str, joinsep: str):
    """Return (dest, func, cols) for a string/row-function formula."""
    dest, func_name, cols, _body = _parse_formula(formula)

    if func_name == "translate":
        trs = str.maketrans(*cols[1:])
        return dest, partial(lambda x: x.translate(trs)), cols[0:1]
    if func_name == "len":
        return dest, partial(lambda x: len(str(x))), cols
    if func_name == "join":
        return dest, partial(lambda x: joinsep.join(str(v) for v in x)), cols
    if func_name == "lower":
        return dest, partial(lambda x: str(x).lower()), cols
    if func_name == "upper":
        return dest, partial(lambda x: str(x).upper()), cols
    if func_name == "zfill":
        return dest, partial(lambda x: str(x).zfill(int(cols[1]))), cols[0:1]
    if func_name == "find":
        return dest, partial(lambda x: str(x).find(str(cols[1]))), cols[0:1]
    if func_name == "match":
        return dest, partial(lambda x: str(x[0]).find(str(x[1]))), cols
    if func_name == "substr":
        start, length = int(cols[1]), int(cols[2])
        return dest, partial(lambda x: str(x)[start: start + length]), cols[0:1]
    if func_name == "substring":
        length = int(cols[2])
        return dest, partial(lambda x: str(x[0])[x[1]: x[1] + length]), cols[0:2]
    if func_name == "replace":
        return dest, partial(lambda x: str(x).replace(cols[1], cols[2])), cols[0:1]

    if func_name == "leftsplit":
        sep, idx = cols[1], int(cols[2])
        def _leftsplit(x):
            s = str(x)
            if sep in s:
                parts = s.split(sep, 1)
                return parts[0] if idx == 0 else parts[1]
            return s if idx == 0 else ""
        return dest, _leftsplit, cols[0:1]

    if func_name == "rightsplit":
        sep, idx = cols[1], int(cols[2])
        def _rightsplit(x):
            s = str(x)
            if sep in s:
                parts = s.rsplit(sep, 1)
                return parts[0] if idx == 0 else parts[1]
            return s if idx == 0 else ""
        return dest, _rightsplit, cols[0:1]

    if func_name == "levenshtein":
        if not _HAS_LEVENSHTEIN:
            raise ImportError(
                "levenshtein() requires the python-Levenshtein package. "
                "Install it with: uv add python-Levenshtein"
            )
        return dest, partial(lambda x: _lev.distance(str(x[0]), str(x[1]))), cols[0:2]

    if func_name == "sort":
        return dest, partial(lambda x: "".join(sorted(str(x)))), cols
    if func_name == "ord":
        def _myord(x):
            try:
                return min(ord(c) for c in str(x))
            except (TypeError, ValueError):
                return 0
        return dest, _myord, cols[0:1]
    if func_name == "template":
        return dest, partial(lambda x: _body.format(**x)), []
    if func_name == "md5":
        return dest, partial(lambda x: hashlib.md5(str(x).encode()).hexdigest()), cols[0:1]

    # Path operations
    if func_name == "basename":
        return dest, partial(lambda x: os.path.basename(str(x))), cols
    if func_name == "stem":
        return dest, partial(lambda x: Path(str(x)).stem), cols
    if func_name == "realpath":
        return dest, partial(lambda x: str(Path(str(x)).resolve())), cols
    if func_name == "isfile":
        return dest, partial(lambda x: int(os.path.isfile(str(x)))), cols
    if func_name == "exists":
        return dest, partial(lambda x: int(os.path.exists(str(x)))), cols

    # DNA / sequence operations
    if func_name == "gccontent":
        return dest, partial(lambda x: _seq.gc_content(x)), cols
    if func_name == "basecount":
        return dest, partial(lambda x: _seq.letter_total(x)), cols
    if func_name == "motifcount":
        pattern = cols[0]
        return dest, partial(lambda x: _seq.count_motif(pattern, x)), cols[1:]
    if func_name == "readoffset":
        return dest, partial(lambda x: _seq.read_offset(*x)), cols

    raise ValueError(f"String function {func_name!r} is not available. "
                     "See dfstat eval --help.")


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------


def _eval(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    joinsep: str = getattr(args, "joinsep", "")
    intbool: bool = getattr(args, "intbool", False)
    existing = set(df.columns)

    # -c: constant columns
    for expr in getattr(args, "constant", []):
        colname, constval_str = (x.strip() for x in expr.split("=", 1))
        try:
            constval: int | float | str = int(constval_str)
        except ValueError:
            try:
                constval = float(constval_str)
            except ValueError:
                constval = constval_str
        df[colname] = constval

    # -f: pandas eval with special-function fallback
    for formula in getattr(args, "formula", []):
        try:
            df.eval(formula, inplace=True)
        except (ValueError, SyntaxError):
            logger.debug(f"df.eval failed for {formula!r}, trying special_function")
            dest, func, cols = _special_function(formula, list(df.columns))
            if len(df) == 0:
                df[dest] = True
            elif isinstance(func, _WhereFunc):
                df[dest] = func(df)
            elif isinstance(func, _FrameFunc):
                # Frame-level functions (idxmax/idxmin) need column labels
                df[dest] = func(df[cols])
            elif len(cols) == 1:
                # Element-wise on a single column (e.g. np.sign, os.path.*)
                df[dest] = df[cols[0]].apply(func)
            else:
                # Multi-column row function: pass numpy array so positional
                # indexing (row[0], row[1], ...) works in pandas 2.x
                df[dest] = df[cols].apply(lambda row: func(row.to_numpy()), axis=1)
        except Exception:
            if len(df) == 0:
                dest = formula.split("=", 1)[0].strip()
                df[dest] = True
            else:
                logger.error(f"Could not evaluate formula: {formula!r}")
                raise

    # -s: string / row functions
    for strfunc in getattr(args, "strfunc", []):
        try:
            dest, func, cols = _string_function(strfunc, joinsep)
            if len(df) == 0:
                df[dest] = True
            elif len(cols) == 0:
                df[dest] = df.apply(func, axis=1)
            elif len(cols) == 1:
                df[dest] = df[cols[0]].apply(func)
            else:
                df[dest] = df[cols].apply(func, axis=1)
        except Exception:
            if len(df) == 0:
                dest = strfunc.split("=", 1)[0].strip()
                df[dest] = True
            else:
                logger.error(f"Could not evaluate string function: {strfunc!r}")
                raise

    # --intbool: convert newly created boolean columns to int
    if intbool:
        new_cols = set(df.columns) - existing
        for col in new_cols:
            if df[col].dtype == np.bool_:
                df[col] = df[col].astype(np.int64)

    return df


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class EvalCommand(BaseCommand):

    @property
    def name(self) -> str:
        return "eval"

    @property
    def help(self) -> str:
        return "Evaluate formulas, constants, and string functions on a DataFrame"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap
        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        io.parser_read(parser)

        g = parser.add_argument_group("formula evaluation")
        g.add_argument(
            "-f", "--formula",
            help=(
                "pandas eval expression, or special function call "
                "(dest = func(col1, col2, ...)).  Repeatable."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "-c", "--constant",
            help=(
                "Add a constant column (dest = value).  "
                "Value is coerced to numeric when possible.  Repeatable."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "-s", "--strfunc",
            help=(
                "String or row function (dest = func(col1, col2, ...)).  "
                "Repeatable.  See --help for full function list."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "--joinsep",
            help="Separator used by the join() string function (default: empty string).",
            default="",
            metavar="SEP",
        )
        g.add_argument(
            "--intbool",
            help="Convert newly created boolean columns to int (False=0, True=1).",
            action="store_true",
        )

        io.parser_output(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        df = _eval(df, args)
        io.printdf(df, args)
