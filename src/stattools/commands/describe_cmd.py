"""
stattools.commands.describe_cmd — dfstat describe subcommand.

Profiles each column in a DataFrame: data type, basic statistics, missing
values, and automatically generated quality/distribution notes.

Output columns (one row per input column)
------------------------------------------
  name, type, count, missing_pct, n_unique,
  mean, std, min, p25, median, p75, max, skew, kurtosis,
  top, top_freq_pct, notes

Numeric columns populate mean…kurtosis; top/top_freq_pct are blank.
Categorical/boolean columns populate top/top_freq_pct; numeric stats are blank.

Notes flags
-----------
  all_missing       — column has no non-null values
  high_missing      — > 20 % missing
  constant          — only one distinct value
  near_constant     — dominant value covers > 95 % of non-null rows
  possible_id       — every non-null value is unique (n_unique == n_rows)
  high_cardinality  — categorical with > 50 distinct values
  outliers_iqr      — at least one value outside 1.5 × IQR fences
  approx_normal     — |skew| < 0.5 and |excess_kurtosis| < 1
  right_skewed      — skew > 1
  left_skewed       — skew < -1
  heavy_tailed      — excess kurtosis > 3
  bimodal_hint      — bimodality coefficient > 0.555 (Pfister et al.)

Optional flags
--------------
  --summary       Print a human-readable summary paragraph to stderr.
  --correlations  Include pairwise correlation analysis in the summary.
  --corr-threshold  Minimum |r| to report (default: 0.7).

Example
-------
    dfstat describe data.tsv
    dfstat describe data.tsv --summary --correlations
"""

import argparse
import sys

import numpy as np
import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io

# ---------------------------------------------------------------------------
# Column type detection
# ---------------------------------------------------------------------------


def _col_type(col: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(col):
        return "boolean"
    if pd.api.types.is_numeric_dtype(col):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"
    return "categorical"


# ---------------------------------------------------------------------------
# Single-column profiling
# ---------------------------------------------------------------------------


def _profile_col(col: pd.Series, n_rows: int) -> dict:
    """Return a dict describing one column."""
    col_type = _col_type(col)
    n_missing = int(col.isna().sum())
    missing_pct = round(100.0 * n_missing / n_rows, 1) if n_rows > 0 else 0.0
    n_valid = n_rows - n_missing
    n_unique = int(col.nunique())

    row: dict = dict(
        name=col.name,
        type=col_type,
        count=n_valid,
        missing_pct=missing_pct,
        n_unique=n_unique,
        mean=np.nan,
        std=np.nan,
        min=np.nan,
        p25=np.nan,
        median=np.nan,
        p75=np.nan,
        max=np.nan,
        skew=np.nan,
        kurtosis=np.nan,
        top="",
        top_freq_pct=np.nan,
        notes="",
    )

    notes: list[str] = []

    if missing_pct > 20.0:
        notes.append("high_missing")
    if missing_pct == 100.0:
        notes.append("all_missing")
        row["notes"] = ",".join(notes)
        return row

    if n_unique == 0:
        row["notes"] = ",".join(notes)
        return row

    if n_unique == 1:
        notes.append("constant")

    if n_unique == n_rows and n_missing == 0:
        notes.append("possible_id")

    if col_type == "numeric":
        vals = col.dropna()

        row["mean"] = float(vals.mean())
        row["std"] = float(vals.std())
        row["min"] = float(vals.min())
        row["p25"] = float(vals.quantile(0.25))
        row["median"] = float(vals.median())
        row["p75"] = float(vals.quantile(0.75))
        row["max"] = float(vals.max())

        if len(vals) >= 4:
            skew = float(vals.skew())
            kurt = float(vals.kurtosis())  # excess kurtosis (normal = 0)
            row["skew"] = round(skew, 4)
            row["kurtosis"] = round(kurt, 4)

            # IQR outlier detection
            q1, q3 = row["p25"], row["p75"]
            iqr = q3 - q1
            if (
                iqr > 0
                and int(((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)).sum()) > 0
            ):
                notes.append("outliers_iqr")

            # Distribution shape
            if abs(skew) < 0.5 and abs(kurt) < 1.0:
                notes.append("approx_normal")
            else:
                if skew > 1.0:
                    notes.append("right_skewed")
                elif skew < -1.0:
                    notes.append("left_skewed")
                if kurt > 3.0:
                    notes.append("heavy_tailed")

                # Bimodality coefficient (Pfister et al. 2013)
                n = len(vals)
                if n > 5:
                    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
                    if abs(denom) > 1e-9 and (skew**2 + 1.0) / denom > 0.555:
                        notes.append("bimodal_hint")

    else:
        vc = col.value_counts()
        if len(vc) > 0:
            row["top"] = str(vc.index[0])
            row["top_freq_pct"] = (
                round(100.0 * int(vc.iloc[0]) / n_valid, 1) if n_valid > 0 else 0.0
            )

            if n_unique > 1 and row["top_freq_pct"] > 95.0:
                notes.append("near_constant")

        if col_type == "categorical" and n_unique > 50:
            notes.append("high_cardinality")

    row["notes"] = ",".join(notes)
    return row


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------


def _correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[tuple]:
    """Return (col_a, col_b, r) pairs where |r| >= threshold, sorted by |r|."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr()
    pairs = []
    for i, a in enumerate(num_cols):
        for b in num_cols[i + 1 :]:
            r = corr.loc[a, b]
            if not np.isnan(r) and abs(r) >= threshold:
                pairs.append((a, b, round(float(r), 4)))
    pairs.sort(key=lambda x: -abs(x[2]))
    return pairs


# ---------------------------------------------------------------------------
# Human-readable summary
# ---------------------------------------------------------------------------


def _flag_names(profile: pd.DataFrame, flag: str) -> list[str]:
    return [str(r["name"]) for _, r in profile.iterrows() if flag in str(r["notes"])]


def _generate_summary(
    profile: pd.DataFrame,
    corr_pairs: list[tuple],
    n_rows: int,
    n_cols: int,
) -> str:
    lines = [f"Dataset: {n_rows} rows × {n_cols} columns"]

    missing = profile[profile["missing_pct"] > 0].sort_values(
        "missing_pct", ascending=False
    )
    if len(missing):
        parts = [f"'{r['name']}' ({r['missing_pct']}%)" for _, r in missing.iterrows()]
        lines.append("Missing data: " + ", ".join(parts))

    for flag, label in [
        ("constant", "Constant columns"),
        ("near_constant", "Near-constant columns"),
        ("possible_id", "Possible identifiers (all unique)"),
        ("high_cardinality", "High cardinality"),
    ]:
        cols = _flag_names(profile, flag)
        if cols:
            lines.append(f"{label}: " + ", ".join(f"'{c}'" for c in cols))

    dist_parts = []
    for _, r in profile[profile["type"] == "numeric"].iterrows():
        n = str(r["notes"])
        tags = []
        if "approx_normal" in n:
            tags.append("approx. normal")
        if "right_skewed" in n:
            tags.append("right-skewed")
        if "left_skewed" in n:
            tags.append("left-skewed")
        if "heavy_tailed" in n:
            tags.append("heavy-tailed")
        if "bimodal_hint" in n:
            tags.append("possibly bimodal")
        if "outliers_iqr" in n:
            tags.append("has outliers (IQR)")
        if tags:
            dist_parts.append(f"'{r['name']}': {', '.join(tags)}")
    if dist_parts:
        lines.append("Distributions: " + "; ".join(dist_parts))

    if corr_pairs:
        parts = [f"'{a}' ↔ '{b}' (r={r})" for a, b, r in corr_pairs]
        lines.append("High correlations: " + ", ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class DescribeCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "describe"

    @property
    def help(self) -> str:
        return "Profile DataFrame columns: types, distributions, and quality flags"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        import argparse as ap

        parser.formatter_class = ap.RawDescriptionHelpFormatter
        parser.epilog = """\
OUTPUT COLUMNS
--------------
One row per input column:

  name           column name
  type           numeric | categorical | boolean | datetime
  count          number of non-null values
  missing_pct    percentage of null values
  n_unique       number of distinct values
  mean std       (numeric only)
  min p25 median p75 max
  skew kurtosis  excess kurtosis (normal distribution = 0)
  top            most frequent value (categorical/boolean)
  top_freq_pct   frequency of top value as a percentage
  notes          comma-separated quality/distribution flags (see below)


NOTES FLAGS
-----------
  all_missing       no non-null values
  high_missing      > 20 % of values are null
  constant          only one distinct value
  near_constant     dominant value covers > 95 % of non-null rows
  possible_id       every non-null value is unique (likely an identifier)
  high_cardinality  categorical column with > 50 distinct values
  outliers_iqr      at least one value outside the 1.5 × IQR fences
  approx_normal     |skew| < 0.5 and |excess kurtosis| < 1
  right_skewed      skew > 1
  left_skewed       skew < -1
  heavy_tailed      excess kurtosis > 3
  bimodal_hint      bimodality coefficient > 0.555 (Pfister et al. 2013)


EXAMPLES
--------
  dfstat describe data.tsv
  dfstat describe data.tsv --summary --correlations
  dfstat describe data.tsv --summary --correlations --corr-threshold 0.5
  dfstat dataset iris | dfstat describe - --summary --correlations
"""
        io.parser_read(parser)

        g = parser.add_argument_group("describe options")
        g.add_argument(
            "--summary",
            help="Print a human-readable summary paragraph to stderr.",
            action="store_true",
        )
        g.add_argument(
            "--correlations",
            help="Compute pairwise correlations and include them in --summary output.",
            action="store_true",
        )
        g.add_argument(
            "--corr-threshold",
            help="Minimum |r| to report as a high correlation (default: 0.7).",
            type=float,
            default=0.7,
            metavar="FLOAT",
        )

        io.parser_output(parser)

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)
        n_rows, n_cols = df.shape

        profile = pd.DataFrame([_profile_col(df[c], n_rows) for c in df.columns])

        corr_pairs: list[tuple] = []
        if getattr(args, "correlations", False):
            threshold = getattr(args, "corr_threshold", 0.7)
            corr_pairs = _correlations(df, threshold)

        if getattr(args, "summary", False) or corr_pairs:
            print(
                _generate_summary(profile, corr_pairs, n_rows, n_cols), file=sys.stderr
            )

        io.printdf(profile, args)
