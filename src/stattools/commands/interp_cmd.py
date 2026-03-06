"""
stattools.commands.interp_cmd — dfstat interp subcommand.

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

  dfstat interp samples.tsv --ref stdcurve.tsv -x fluorescence -v concentration

Different x-column names in each file:

  dfstat interp samples.tsv --ref stdcurve.tsv \\
      -x sample_fluor --refx std_fluor -v concentration

Interpolate multiple y-columns at once:

  dfstat interp data.tsv --ref ref.tsv -x pos -v y1 y2 -d col1 col2

Per-chromosome genetic-map lookup (grouped):

  dfstat interp intervals.tsv --ref genmap.tsv \\
      -x bp_mid --refx pos_bp -v pos_cM -d cM_mid -g chrom
"""

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io, check_cols


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

    if fill == "edge":
        fill_value = "extrapolate"  # use boundary values — interp1d handles this
        # Actually for "edge" clamping we use fill_value=(y[0], y[-1])
        fill_value = (None, None)  # set per-column below

    for src, dest in zip(val_cols, dest_cols):
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
    groupcols: Optional[list[str]],
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
                    ref_groups[grp_key], grp_data,
                    x_ref, x_query, val_cols, dest_cols, method, fill,
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
            "-x", "--xcol",
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
            "-v", "--val",
            required=True,
            nargs="+",
            metavar="COL",
            help="Y-axis column(s) in the reference file to interpolate.",
        )
        g.add_argument(
            "-d", "--destcol",
            default=None,
            nargs="+",
            metavar="NAME",
            help="Output column name(s) in DATAFILE (default: same as -v/--val).",
        )
        g.add_argument(
            "-g", "--groupcol",
            nargs="+",
            default=None,
            metavar="COL",
            help="Interpolate independently within each group defined by these column(s). "
                 "Must be present in both DATAFILE and the reference file.",
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
            help="Out-of-range fill strategy: nan (default) or edge (clamp to boundary).",
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
                f"--destcol has {len(dest_cols)} name(s) but --val has {len(args.val)} column(s). "
                "They must match."
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
