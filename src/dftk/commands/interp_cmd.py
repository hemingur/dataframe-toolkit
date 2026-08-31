"""
dftk.commands.interp_cmd — dftk interp subcommand.

Enriches a data table by interpolating values from a reference curve.

The reference file (--ref) defines a 1-D function: a sorted x-column and one
or more y-columns.  For each row in DATAFILE the command looks up the x-value
in the reference and interpolates the corresponding y-value(s), appending them
as new column(s).

Grouping (-g) can be used when both files contain a grouping column (e.g. a
chromosome or sample ID): interpolation is then performed independently within
each matching group.

INTERPOLATION METHODS (--method)
---------------------------------
  linear   (default) Piecewise-linear interpolation.
  nearest  Nearest-neighbour (step function).
  cubic    Cubic spline (requires >= 4 reference points per group).
  zero     Zero-order hold (previous value).
  slinear  Linear in the B-spline sense.
  quadratic / cubic  Higher-order B-splines.

Out-of-range behaviour (--fill)
--------------------------------
  nan    (default) Values outside the reference x-range are NaN.
  edge   Clamp to the boundary values of the reference curve.

KNOWN LIMITATION
-----------------
When the reference curve has a flat segment (zero slope, e.g. a recombination-
cold spot in a genetic map), interpolation returns the left-boundary value for
all query points within that segment.  Round-trip conversions that pass through
a flat segment require manual handling — this command does not attempt to
correct for it.

EXAMPLES
--------
Basic interpolation — add "concentration" to samples by interpolating a
standard curve:

  dftk interp samples.tsv --ref stdcurve.tsv -x fluorescence -v concentration

Runnable version: split the bundled `sunspots` time series (YEAR,
SUNACTIVITY) into even and odd years, treat the even years as a dense
reference curve, and interpolate SUNACTIVITY at the odd years from it:

  dftk dataset sunspots -o \\
      | dftk query ... --sql \\
          "SELECT * FROM data WHERE CAST(YEAR AS INT) % 2 = 0 ORDER BY YEAR" \\
          -o sunspots_even.tsv
  dftk dataset sunspots -o \\
      | dftk query ... --sql \\
          "SELECT * FROM data WHERE CAST(YEAR AS INT) % 2 = 1 ORDER BY YEAR" \\
          -o sunspots_odd.tsv
  dftk interp sunspots_odd.tsv --ref sunspots_even.tsv \\
      -x YEAR -v SUNACTIVITY -d sunactivity_interp

Different x-column names in each file (rename YEAR to yr_ref in a copy of
the reference file):

  dftk query sunspots_even.tsv --sql "SELECT YEAR AS yr_ref, SUNACTIVITY FROM data" \\
      -o sunspots_even_renamed.tsv
  dftk interp sunspots_odd.tsv --ref sunspots_even_renamed.tsv \\
      -x YEAR --refx yr_ref -v SUNACTIVITY -d sunactivity_interp

Interpolate multiple y-columns at once, grouped — build a per-cut
carat-to-(price, depth) reference curve from the bundled `diamonds`
dataset, then look up each row's *expected* price and depth from its own
cut's curve at its exact carat, for comparison against the actual values:

  dftk dataset diamonds -o \\
      | dftk binx ... -c carat -b 0:5:0.1 -d carat_bin --usevalue m -o \\
      | dftk pivot ... -v price depth -i carat_bin cut -f mean -o \\
      | dftk query ... --sql "SELECT * FROM data ORDER BY cut, carat_bin" \\
          -o diamonds_ref.tsv
  dftk dataset diamonds -o \\
      | dftk interp ... --ref diamonds_ref.tsv -x carat --refx carat_bin \\
          -v mean_price mean_depth -d expected_price expected_depth -g cut
"""

import argparse

import numpy as np
import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import check_cols, io

# ---------------------------------------------------------------------------
# Core interpolation
# ---------------------------------------------------------------------------


def _interp_one(
    ref: pd.DataFrame,
    query: pd.DataFrame,
    x_ref: str,
    x_query: str,
    val_cols: list[str],
    dest_cols: list[str],
    method: str,
    fill: str,
) -> pd.DataFrame:
    """Interpolate *val_cols* from *ref* at *x_query* positions in *query*.

    Returns a copy of *query* with *dest_cols* added.
    """
    import scipy.interpolate

    query = query.copy()
    x_new = query[x_query].to_numpy(dtype=float)

    for src, dest in zip(val_cols, dest_cols, strict=False):
        x_ref_arr = ref[x_ref].to_numpy(dtype=float)
        y_ref_arr = ref[src].to_numpy(dtype=float)

        if fill == "edge":
            fv = (y_ref_arr[0], y_ref_arr[-1])
        else:
            fv = np.nan

        fx = scipy.interpolate.interp1d(
            x_ref_arr,
            y_ref_arr,
            kind=method,
            bounds_error=False,
            fill_value=fv,
            assume_sorted=True,
        )
        query[dest] = fx(x_new)

    return query


def _interp(
    ref: pd.DataFrame,
    data: pd.DataFrame,
    x_ref: str,
    x_query: str,
    val_cols: list[str],
    dest_cols: list[str],
    method: str,
    fill: str,
    groupcols: list[str] | None,
) -> pd.DataFrame:
    """Run interpolation, optionally within groups."""

    if groupcols:
        parts = []
        # Both ref and data are grouped with the same groupcols, so keys match directly
        ref_groups = {k: v for k, v in ref.groupby(groupcols)}
        for grp_key, grp_data in data.groupby(groupcols, sort=False):
            if grp_key not in ref_groups:
                # Group present in data but not in ref — fill with NaN
                grp_data = grp_data.copy()
                for dest in dest_cols:
                    grp_data[dest] = np.nan
                parts.append(grp_data)
                continue
            parts.append(
                _interp_one(
                    ref_groups[grp_key],
                    grp_data,
                    x_ref,
                    x_query,
                    val_cols,
                    dest_cols,
                    method,
                    fill,
                )
            )
        return pd.concat(parts, ignore_index=True)

    return _interp_one(ref, data, x_ref, x_query, val_cols, dest_cols, method, fill)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------


class InterpCommand(BaseCommand):
    name = "interp"
    help = "Enrich a table by interpolating values from a reference curve."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = __doc__

        self.add_io_arguments(parser)

        g = parser.add_argument_group("interpolation options")
        g.add_argument(
            "--ref",
            required=True,
            metavar="FILE",
            help="Reference file containing the curve to interpolate from.",
        )
        g.add_argument(
            "-x",
            "--xcol",
            required=True,
            metavar="COL",
            help="X-axis column in DATAFILE (the query positions).",
        )
        g.add_argument(
            "--refx",
            default=None,
            metavar="COL",
            help="X-axis column in the reference file (default: same as -x/--xcol).",
        )
        g.add_argument(
            "-v",
            "--val",
            required=True,
            nargs="+",
            metavar="COL",
            help="Y-axis column(s) in the reference file to interpolate.",
        )
        g.add_argument(
            "-d",
            "--destcol",
            default=None,
            nargs="+",
            metavar="NAME",
            help="Output column name(s) in DATAFILE (default: same as -v/--val).",
        )
        g.add_argument(
            "-g",
            "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help=(
                "Interpolate independently within each group. "
                "Must be present in both DATAFILE and the reference file."
            ),
        )
        g.add_argument(
            "--method",
            default="linear",
            metavar="METHOD",
            choices=["linear", "nearest", "zero", "slinear", "quadratic", "cubic"],
            help="Interpolation method (default: linear).",
        )
        g.add_argument(
            "--fill",
            default="nan",
            choices=["nan", "edge"],
            help="Out-of-range fill: nan (default) or edge (clamp to boundary).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        # Read the data file (DATAFILE positional arg, handled by io.read)
        data = io.read(args)

        # Read the reference file using the same read settings
        import copy

        ref_args = copy.copy(args)
        ref_args.DATAFILE = args.ref
        ref = io.read(ref_args)

        x_ref = args.refx if args.refx is not None else args.xcol
        dest_cols = args.destcol if args.destcol is not None else list(args.val)

        if len(dest_cols) != len(args.val):
            raise ValueError(
                f"--destcol has {len(dest_cols)} name(s) but "
                f"--val has {len(args.val)} column(s). They must match."
            )

        # Validate columns
        check_cols(data, [args.xcol], "-x/--xcol")
        check_cols(data, args.groupcol, "-g/--groupcol")
        check_cols(ref, [x_ref], "--refx")
        check_cols(ref, args.val, "-v/--val")
        if args.groupcol:
            check_cols(ref, args.groupcol, "-g/--groupcol (in ref file)")

        result = _interp(
            ref=ref,
            data=data,
            x_ref=x_ref,
            x_query=args.xcol,
            val_cols=list(args.val),
            dest_cols=dest_cols,
            method=args.method,
            fill=args.fill,
            groupcols=args.groupcol,
        )

        io.printdf(result, args)
