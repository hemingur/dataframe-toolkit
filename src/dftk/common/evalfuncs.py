"""
dftk.common.evalfuncs — the functions behind dftk eval's -f and -s options.

-f: a formula whose right-hand side is a single call to one of
SPECIAL_FUNCTIONS (row sum/mean, where, sign, statistics, scipy.special,
path functions) runs that function; anything else goes to pandas df.eval().
-s: string / row functions (string_function).
"""

import fnmatch
import hashlib
import os
import re
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from dftk.common import seq as _seq

# Optional dependency: python-Levenshtein
try:
    import Levenshtein as _lev

    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False


class FormulaMistake(ValueError):
    """A mistake in a formula whatever the column types (an unknown name,
    bad syntax, no assignment): reported even on an empty table."""


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------


def parse_formula(formula: str) -> tuple[str, str, list[str], str]:
    """Parse 'dest = func(col1, col2, ...)' into (dest, func, cols, body).

    Returns the raw argument string as *body* so callers can use it as a
    glob pattern (e.g. for colsum).
    """
    try:
        dest, right = (x.strip() for x in formula.split("=", 1))
        func, right = (x.strip() for x in right.split("(", 1))
        body, _ = (x.strip() for x in right.rsplit(")", 1))
    except Exception:
        raise FormulaMistake(
            f"Bad formula syntax: {formula!r}  Expected: dest = func(col1, col2, ...)"
        ) from None
    cols = [x.strip() for x in body.split(",")]
    return dest, func, cols, body


# ---------------------------------------------------------------------------
# Special functions (-f)
# ---------------------------------------------------------------------------

# The function tables below hold names, not function objects: dftk.common.stats
# and scipy.special are imported only when a formula actually uses one of them,
# so plain `dftk eval -f` formulas don't pay scipy's import time.

# dftk.common.stats functions, same name in the formula as in the module
_STATS_DISPATCH = {
    "binom_test",
    "fisher_test",
    "fisher_OR",
    "boschloo_test",
    "boschloo_OR",
    "pval2se",
    "t2pval",
    "chi2_to_neglogp",
    "neglogp_to_chi2",
    "pval_to_chi2",
    "generalized_poisson_nll",
}

_PATH_FUNCS = {"basename", "dirname", "exists", "getsize", "realpath"}
_NP_FUNCS = {"sign", "cbrt", "log1p", "expm1"}
_ROW_AGG_FUNCS = {"sum", "mean", "std", "min", "max", "median"}
_ROW_IDX_FUNCS = {"idxmax", "idxmin"}

# scipy.special single-column functions (element-wise, vectorised):
# formula name -> scipy.special attribute name
_SCIPY_SINGLE = {
    "gamma": "gamma",  # Γ(x)
    "lgamma": "gammaln",  # log Γ(x)
    "digamma": "digamma",  # ψ(x)  (log-derivative of Γ)
    "erf": "erf",  # error function
    "erfc": "erfc",  # complementary error function
    "erfinv": "erfinv",  # inverse error function
    "expit": "expit",  # logistic sigmoid  1/(1+exp(-x))
    "logit": "logit",  # log(x/(1-x))
    # Bessel J (first kind), fixed order
    "j0": "j0",
    "j1": "j1",
    # Bessel Y (second kind), fixed order
    "y0": "y0",
    "y1": "y1",
    # Modified Bessel I, fixed order
    "i0": "i0",
    "i1": "i1",
    # Modified Bessel K, fixed order
    "k0": "k0",
    "k1": "k1",
}

# scipy.special two-column functions: func(col_a, col_b)
_SCIPY_DUAL = {"beta", "betaln"}  # B(a, b), log B(a, b)

# scipy.special parametric Bessel: func_name(order_literal, col)
# Covers all-order variants: jn(n,x), yn(n,x), iv(v,x), kv(v,x)
_SCIPY_BESSEL_ORDER = {"jn", "yn", "iv", "kv"}


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
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
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


SPECIAL_FUNCTIONS = (
    {
        "sum",
        "mean",
        "std",
        "min",
        "max",
        "median",
        "overlap",
        "applymask",
        "colsum",
        "where",
    }
    | _ROW_IDX_FUNCS
    | _PATH_FUNCS
    | _NP_FUNCS
    | set(_STATS_DISPATCH)
    | set(_SCIPY_SINGLE)
    | set(_SCIPY_DUAL)
    | _SCIPY_BESSEL_ORDER
)


# Also built into pandas df.eval, which handles them faster: leave them to it
_PANDAS_FUNCS = {"log1p", "expm1"}

# "dest = name(" at the start of a formula
_CALL = re.compile(r"\s*[A-Za-z_]\w*\s*=\s*([A-Za-z_]\w*)\s*\(")


def is_special_call(formula: str) -> bool:
    """Return True if *formula* is 'dest = name(...)' with a special function.

    The call must be the whole right-hand side, so 'z = sum(a, b) * 2' is
    not one: pandas then reports that sum can't be used inside an expression.
    """
    m = _CALL.match(formula)
    if not m or m.group(1) not in SPECIAL_FUNCTIONS or m.group(1) in _PANDAS_FUNCS:
        return False
    rest = formula[m.end() - 1 :].rstrip()
    depth = 0
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i == len(rest) - 1
    return False


def special_function(formula: str, df_columns: list[str]):
    """Return (dest, func, cols) for a special-function formula."""
    dest, func_name, cols, body = parse_formula(formula)

    if func_name not in SPECIAL_FUNCTIONS:
        raise FormulaMistake(
            f"Function {func_name!r} is not available as a "
            "special function.  See dftk eval --help."
        )

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
        from dftk.common import stats as _stats

        func = getattr(_stats, func_name)
    elif func_name in _SCIPY_SINGLE:
        from scipy import special as _sp

        func = getattr(_sp, _SCIPY_SINGLE[func_name])
    elif func_name in _SCIPY_DUAL:
        from scipy import special as _sp

        dual = getattr(_sp, func_name)
        func = partial(lambda x: dual(x[0], x[1]))
    elif func_name in _SCIPY_BESSEL_ORDER:
        from scipy import special as _sp

        try:
            order = float(cols[0])
        except (ValueError, IndexError):
            raise ValueError(
                f"{func_name}() first argument must be a numeric order literal, "
                f"e.g.  result = {func_name}(2, xcol)"
            ) from None
        func = partial(getattr(_sp, func_name), order)
        cols = cols[1:]
    else:
        func = func_name  # bare string passed to pandas agg

    return dest, func, cols


# ---------------------------------------------------------------------------
# String / row functions (used by -s)
# ---------------------------------------------------------------------------


def string_function(formula: str, joinsep: str):
    """Return (dest, func, cols) for a string/row-function formula."""
    dest, func_name, cols, _body = parse_formula(formula)

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
        return dest, partial(lambda x: str(x)[start : start + length]), cols[0:1]
    if func_name == "substring":
        length = int(cols[2])
        return dest, partial(lambda x: str(x[0])[x[1] : x[1] + length]), cols[0:2]
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
        return (
            dest,
            partial(lambda x: hashlib.md5(str(x).encode()).hexdigest()),
            cols[0:1],
        )

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
        return dest, partial(lambda x: os.path.exists(str(x))), cols

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

    raise ValueError(
        f"String function {func_name!r} is not available. See dftk eval --help."
    )


def apply_special_function(df: pd.DataFrame, formula: str) -> None:
    """Evaluate a special-function formula and store its column in *df*."""
    dest, func, cols = special_function(formula, list(df.columns))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        hint = ""
        if any(re.fullmatch(r"[-+]?[\d.]+(e[-+]?\d+)?", c) for c in missing):
            hint = (
                "  For a fixed value, add it as a column first: "
                "-c 'p = 0.5' -f 'z = func(x, p)'."
            )
        raise FormulaMistake(f"-f {formula!r}: column(s) not found: {missing}.{hint}")
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
        df[dest] = df[cols].apply(lambda row, f=func: f(row.to_numpy()), axis=1)
