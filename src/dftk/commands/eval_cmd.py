"""
dftk.commands.eval_cmd — dftk eval subcommand.

Evaluates formulas, constants, and string/row functions against a DataFrame.
Three formula types are supported, each with its own flag:

  -f  pandas eval expressions  (df.eval, falls back to special_function)
  -c  constant column assignments
  -s  string / row functions
"""

import argparse
import logging
import re

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.evalfuncs import (
    SPECIAL_FUNCTIONS,
    FormulaMistake,
    apply_special_function,
    is_special_call,
    string_function,
)
from dftk.common.io import io

logger = logging.getLogger(__name__)


_EPILOG = """\
FORMULA EXPRESSIONS (-f)
-------------------------
Each -f argument is first passed to pandas df.eval().  This handles
arithmetic and boolean expressions directly on column names:

  dftk eval data.tsv -f "z = x + y * 2"
  dftk dataset tips -o | dftk eval ... -f "bill_and_tip = total_bill + tip * 2"
  dftk dataset tips -o | dftk eval ... -f "good_tip = (tip > 3) & (total_bill < 30)"

Column methods are pandas Series methods, written COL.method(...); text
methods are under .str (name.str.lower(), not name.lower() or tolower()):

  x = x.abs()   x = x.round(1)   x = x.clip(lower=0)   y = name.str.upper()


FILLING MISSING VALUES (-f)
---------------------------
fillna() fills the empty cells of a column; the other cells are kept:

  dftk eval data.tsv -f "x = x.fillna(y)"          from another column
  -f "x = x.fillna(0)"   -f "sex = sex.fillna('unknown')"     a fixed value
  -f "x = x.fillna(y).fillna(0)"                   y first, then 0
  -f "x_filled = x.fillna(y)"                      keep x, add a new column
  -f "missing = x.isna()"                          mark the empty cells

Defaults from another table: a column that is in both files gets the
suffix _r from merge, so fill from it and drop it. Here the defaults are
each species' median body mass:

  dftk dataset penguins -o | dftk pivot ... -i species -v body_mass_g \\
      -f median --move median_body_mass_g:body_mass_g -o defaults.parquet
  dftk dataset penguins -o \\
      | dftk merge ... -r defaults.parquet -k species -t left -o \\
      | dftk eval ... -f "body_mass_g = body_mass_g.fillna(body_mass_g_r)" \\
          --drop body_mass_g_r

The group mean (or 'median', 'min', ...), in one step:

  dftk dataset penguins -o | dftk eval ... \\
      -f "mean = body_mass_g.groupby(species).transform('mean')" \\
      -f "body_mass_g = body_mass_g.fillna(mean)" --drop mean

From neighbouring rows (the co2 dataset is a weekly series with gaps):

  dftk dataset co2 -o | dftk eval ... -f "ffill = co2.ffill()" \\
      -f "bfill = co2.bfill()" -f "linear = co2.interpolate()"

  ffill()          the last value above; the first rows stay empty
  bfill()          the next value below; the last rows stay empty
  ffill(limit=1)   fill at most one row after each value
  interpolate()    a straight line between the values above and below, by
                   row position: rows must be evenly spaced (for x-based
                   lookup from a reference curve, see dftk interp)

Row order matters: these methods use the rows in the order they are in.
Sort first, in a step of its own (--sortasc is applied to the output, so
in the same command it comes after the fill):

  dftk eval data.tsv --sortasc site day -o | dftk eval ... -f "..."

Several series in one table (sites, countries, samples) must be filled
within each group, or values are taken from the neighbouring group:

  -f "x = x.groupby(site).ffill()"
  -f "x = x.groupby(site).bfill()"
  -f "x = x.groupby(site).ffill().fillna(x.groupby(site).bfill())"

A group's rows don't have to be next to each other, but within each group
they must be in order. In healthexp, rows are sorted by Year with the
countries mixed together, so a grouped ffill is right as it is:

  dftk dataset healthexp -o \\
      | dftk eval ... -f "le = Life_Expectancy.where(Year % 10 != 5)" -o \\
      | dftk eval ... -f "le_filled = le.groupby(Country).ffill()"

interpolate() has no grouped form. Sort so that each group's rows are
together; then gaps inside a group are right, but a gap at the start or
end of a group is filled across the boundary. To leave those empty, keep
only values with something before and after them in the same group:

  dftk dataset healthexp -o \\
      | dftk eval ... -f "le = Life_Expectancy.where(Year % 10 != 5)" \\
          --sortasc Country Year -o \\
      | dftk eval ... -f "before = le.groupby(Country).ffill().notna()" \\
          -f "after = le.groupby(Country).bfill().notna()" \\
          -f "le_linear = le.interpolate().where(before & after)" \\
          --drop before after

Only empty cells are filled; missing rows (a year with no row at all) are
not added.

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
        dftk dataset tips -o | dftk eval ... \
          -f "big = total_bill > 30" \
          -f "label = where(big, \"large\", \"small\")"

  Bitwise:
    applymask(col, mask_col)     — col & mask_col
    overlap(lo1, hi1, lo2, hi2)  — length of interval overlap

  Column aggregation (glob pattern):
    colsum(col*)     — sum of all columns matching the glob pattern

  NumPy math (element-wise):
    sign(col)  cbrt(col)  log1p(col)  expm1(col)

  scipy.special — log-gamma / digamma:
    gamma(col)   lgamma(col)   digamma(col)
    beta(a, b)   betaln(a, b)

  scipy.special — error functions:
    erf(col)   erfc(col)   erfinv(col)

  scipy.special — sigmoid / logit:
    expit(col)   logit(col)

  scipy.special — Bessel functions (fixed order):
    j0(col)  j1(col)   — Bessel J, order 0 / 1
    y0(col)  y1(col)   — Bessel Y, order 0 / 1
    i0(col)  i1(col)   — modified Bessel I, order 0 / 1
    k0(col)  k1(col)   — modified Bessel K, order 0 / 1

  scipy.special — Bessel functions (parametric order):
    jn(n, col)   — Bessel J, integer order n
    yn(n, col)   — Bessel Y, integer order n
    iv(v, col)   — modified Bessel I, real order v
    kv(v, col)   — modified Bessel K, real order v
    (n/v is a numeric literal, e.g.  result = jn(3, x))

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

  dftk dataset tips -o | dftk eval ... -c "batch = 3"
  dftk dataset tips -o | dftk eval ... -c "region = northeast"

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
# Core eval logic
# ---------------------------------------------------------------------------


# A message pandas gives for a method that no column has, whatever its dtype
_NO_METHOD = re.compile(r"'Series' object has no attribute '(\w+)'")

# pandas' message for a function it doesn't know
_NO_FUNCTION = re.compile(r'"(\w+)" is not a supported function')


def _formula_error(formula: str, exc: Exception) -> str:
    """Return a short message for a mistake pandas found in an -f formula."""
    msg = str(exc)
    m = _NO_METHOD.search(msg)
    if m:
        name = m.group(1)
        col = re.search(rf"([A-Za-z_]\w*)\.{name}\b", formula)
        col = col.group(1) if col else "COL"
        if hasattr(pd.Series.str, name):
            hint = f"Text methods are under .str: {col}.str.{name}()."
        else:
            hint = f"Text methods are under .str, e.g. {col}.str.lower()."
        return (
            f"-f {formula!r}: column has no method {name!r}. {hint}  See dftk eval -h."
        )
    m = _NO_FUNCTION.search(msg)
    if m:
        name = m.group(1)
        if name in SPECIAL_FUNCTIONS:
            return (
                f"-f {formula!r}: {name}() must be the whole right-hand side, "
                f"e.g. -f 'tmp = {name}(...)' -f 'z = tmp * 2'."
            )
        return f"-f {formula!r}: unknown function {name!r}.  See dftk eval -h."
    return f"-f {formula!r}: {msg}"


def _eval_formula(df: pd.DataFrame, formula: str) -> pd.DataFrame:
    """Evaluate one -f formula: a special function, or else pandas df.eval()."""
    if is_special_call(formula):
        apply_special_function(df, formula)
        return df
    try:
        # Not inplace: that writes into the existing column, which must keep
        # its dtype, so x = x.fillna(y) or x = x.isna() failed whenever the
        # result had another dtype
        result = df.eval(formula)
    except (NameError, SyntaxError) as exc:
        # An unknown column (pandas' UndefinedVariableError) or bad syntax
        raise FormulaMistake(_formula_error(formula, exc)) from exc
    except (AttributeError, TypeError, KeyError, ValueError) as exc:
        # Raised while pandas evaluated the user's expression, so most likely
        # a mistake in it (wrong method, arguments, or .str on numbers)
        msg = str(exc)
        mistake = _NO_METHOD.search(msg) or _NO_FUNCTION.search(msg)
        cls = FormulaMistake if mistake else ValueError
        raise cls(_formula_error(formula, exc)) from exc
    if not isinstance(result, pd.DataFrame):
        # No "dest =": df.eval returned the values instead of a table
        raise FormulaMistake(
            f"-f {formula!r} has no assignment; write it as 'name = {formula}'"
        )
    return result


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

    # -f: special functions, or else pandas df.eval
    for formula in getattr(args, "formula", []):
        try:
            df = _eval_formula(df, formula)
        except Exception as exc:
            # An empty table has no column types (pandas reads them as
            # float64), so on one only mistakes that don't depend on types
            # are reported; for the rest the column is just added
            if len(df) != 0 or isinstance(exc, FormulaMistake):
                raise
            df[formula.split("=", 1)[0].strip()] = True

    # -s: string / row functions
    for strfunc in getattr(args, "strfunc", []):
        try:
            dest, func, cols = string_function(strfunc, joinsep)
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
            "-f",
            "--formula",
            help=(
                "pandas eval expression, or special function call "
                "(dest = func(col1, col2, ...)).  Repeatable."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "-c",
            "--constant",
            help=(
                "Add a constant column (dest = value).  "
                "Value is coerced to numeric when possible.  Repeatable."
            ),
            default=[],
            action="append",
            metavar="EXPR",
        )
        g.add_argument(
            "-s",
            "--strfunc",
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
            help="Separator used by the join() string function (default: empty).",
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
