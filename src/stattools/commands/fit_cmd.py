"""
stattools.commands.fit_cmd — dfstat fit subcommand.

Port of dfsmfit.py: OLS / robust (RLM) / weighted (WLS) linear regression
using statsmodels with R-style formulas.

TODO: variable selection — add --select forward/backward/both to perform
  stepwise AIC/BIC-driven predictor selection on top of OLS.  The selected
  formula would be logged to stderr and the output table would be identical
  to a normal fit run.  Lasso / Ridge / Elastic Net regularisation paths are
  better handled as a separate command (e.g. dfstat regpath) using sklearn,
  since the output and cross-validation workflow differ fundamentally.

TODO: add support for additional model types via a --model flag:
  --model logit / glm (+ --family binomial/poisson/gamma) — logistic and
      generalised linear models via smf.logit() / smf.glm().
  --model mixedlm (+ --groups COL) — mixed linear models with random effects
      via smf.mixedlm(); output table would need extra columns for random
      effects covariance.  res2df() would need a branch for each new type
      since result attributes (rsquared, condnum, etc.) differ.

Output modes
------------
Default           Tidy coefficient table (one row per predictor).
--summary         Print the full statsmodels summary text (no grouping).
--anova           Print the ANOVA table (no grouping, OLS only).

Grouping (-g)
-------------
Runs the regression independently for each group and concatenates the
tidy coefficient tables into one output dataframe.
"""

import logging
import argparse

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.anova as sa

from stattools.commands.base import BaseCommand
from stattools.common.io import io, check_cols

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core regression helpers  (also imported by scale_cmd)
# ---------------------------------------------------------------------------


def standard_model(df: pd.DataFrame, args: argparse.Namespace):
    """Return an OLS or RLM (robust) model, not yet fitted."""
    if getattr(args, "robust", False):
        return sm.robust.robust_linear_model.RLM.from_formula(
            formula=args.formula, data=df, M=sm.robust.norms.HuberT()
        )
    return smf.ols(formula=args.formula, data=df)


def weighted_model(df: pd.DataFrame, args: argparse.Namespace):
    """Return a WLS model, not yet fitted."""
    return sm.regression.linear_model.WLS.from_formula(
        args.formula, df, weights=df[args.weights]
    )


def regress_it(df: pd.DataFrame, args: argparse.Namespace):
    """Fit the appropriate model and return the fitted result object."""
    if getattr(args, "weights", None) is not None:
        mod = weighted_model(df, args)
    else:
        mod = standard_model(df, args)
    return mod.fit()


# ---------------------------------------------------------------------------
# Result → DataFrame
# ---------------------------------------------------------------------------


def res2df(
    res,
    args: argparse.Namespace,
    groupname=None,
) -> pd.DataFrame:
    """Convert a fitted statsmodels result to a tidy coefficient DataFrame."""
    is_robust = getattr(args, "robust", False)

    ci = res.conf_int()
    out_dict = {
        "tag":      args.tag,
        "variable": res.model.data.ynames,
        "marker":   res.model.data.xnames,
        "effect":   res.params,
        "stdev":    res.bse,
        "pval":     res.pvalues,
        "nobs":     res.nobs,
        "cilo":     ci[0],
        "cihi":     ci[1],
    }
    stat_cols = ["tag", "variable", "marker", "effect", "stdev", "pval", "nobs", "cilo", "cihi"]

    if not is_robust:
        out_dict["rsquared"] = res.rsquared
        out_dict["condnum"] = res.condition_number
        stat_cols += ["rsquared", "condnum"]

    if groupname is None:
        return pd.DataFrame(out_dict)[stat_cols]

    # Grouped: prepend group-column values
    vals = list(groupname) if isinstance(groupname, tuple) else [groupname]
    grpcols = args.groupcol[:]
    n = len(res.params)
    for grpcol, val in zip(grpcols, vals):
        out_dict[grpcol] = [val] * n
    return pd.DataFrame(out_dict)[grpcols + stat_cols]


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


_EPILOG = """\
OUTPUT COLUMNS
--------------
  tag       Value of --tag (label for the result set).
  variable  Dependent variable name (left-hand side of formula).
  marker    Predictor name (Intercept plus each right-hand side term).
  effect    Estimated coefficient.
  stdev     Standard error of the coefficient.
  pval      Two-sided p-value.
  nobs      Number of observations used.
  cilo      Lower bound of the 95 % confidence interval.
  cihi      Upper bound of the 95 % confidence interval.
  rsquared  R² (OLS / WLS only).
  condnum   Condition number (OLS / WLS only).


EXAMPLES
--------
Simple OLS:

  dfstat fit data.tsv -f "y ~ x"

Grouped OLS (one regression per group):

  dfstat fit data.tsv -f "y ~ x" -g group

Weighted least squares:

  dfstat fit data.tsv -f "y ~ x" -w weight_col

Robust regression (HuberT):

  dfstat fit data.tsv -f "y ~ x" --robust

Print detailed summary (no grouping):

  dfstat fit data.tsv -f "y ~ x" --summary

ANOVA table (OLS, no grouping):

  dfstat fit data.tsv -f "y ~ x + z" --anova
"""


class FitCommand(BaseCommand):
    name = "fit"
    help = "OLS / robust / weighted linear regression (R-style formula)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        self.add_io_arguments(parser)

        g = parser.add_argument_group("regression options")
        g.add_argument(
            "-f", "--formula",
            required=True,
            metavar="FORMULA",
            help="R-style formula, e.g. 'y ~ x + z'.",
        )
        g.add_argument(
            "-w", "--weights",
            default=None,
            metavar="COL",
            help="Column name to use as observation weights (WLS).",
        )
        g.add_argument(
            "--tag",
            default="tag",
            metavar="LABEL",
            help="Label added to the 'tag' column in every output row (default: tag).",
        )
        g.add_argument(
            "-r", "--robust",
            action="store_true",
            help="Use robust linear regression (RLM with HuberT norm) instead of OLS.",
        )
        g.add_argument(
            "-g", "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="Run a separate regression for each combination of these columns.",
        )
        g.add_argument(
            "-s", "--summary",
            action="store_true",
            help="Print the full statsmodels summary (overrides tabular output; no grouping).",
        )
        g.add_argument(
            "--anova",
            action="store_true",
            help="Print the ANOVA table (OLS only, no grouping).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        check_cols(df, args.groupcol, "-g/--groupcol")
        if args.weights is not None:
            check_cols(df, [args.weights], "-w/--weights")

        if args.groupcol is None:
            res = regress_it(df, args)
            if args.summary:
                print(res.summary())
                return
            if args.anova:
                print(sa.anova_lm(res))
                return
            out = res2df(res, args)
        else:
            grouped = df.groupby(args.groupcol)
            parts = []
            for groupname, groupdf in grouped:
                try:
                    res = regress_it(groupdf, args)
                except Exception as exc:
                    logger.error(
                        "Regression failed for group %s: %s", groupname, exc
                    )
                    raise
                parts.append(res2df(res, args, groupname))
            out = pd.concat(parts, ignore_index=True)

        io.printdf(out, args)
