"""
stattools.common.plot — shared matplotlib infrastructure for dfstat plot commands.

All plot commands (scat, line, hist) import from here for:
  - Figure / axes creation with publication-quality defaults
  - Font size presets (publication / screen / presentation)
  - Figure size presets (single / double / full column, or explicit WxH)
  - Colourblind-friendly default palette (Wong 2011)
  - Shared argument groups for argparse
  - Axis limit helpers
  - Save-or-show helper

Font size hierarchy
-------------------
Each preset maps three semantic roles to point sizes:

  small   → tick label numbers on axes
  medium  → axis labels, legend text
  large   → figure title, legend title

Presets:

  publication   6 / 8 / 9 pt   (Nature / Science figure standards)
  screen        9 / 11 / 13 pt  (default — readable on screen)
  presentation  12 / 14 / 16 pt (slides)

Figure size presets (width × auto-height at golden ratio)
----------------------------------------------------------
  single   3.50" × 2.16"   Nature single column  (89 mm)
  double   7.20" × 4.45"   Nature double column  (183 mm)
  full     7.20" × 4.45"   alias for double
  WxH      explicit inches, e.g. "5x3.5"
  default: 5x3.5

Colour palette
--------------
Wong (2011) colourblind-safe 8-colour palette, also print-safe in greyscale.
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Optional

import matplotlib
import matplotlib.ticker as ticker
import numpy as np
# matplotlib.pyplot is imported lazily (inside functions) so that
# matplotlib.use() can be called in execute() before pyplot initialises
# the backend.  Never add "import matplotlib.pyplot as plt" at module level.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colourblind-safe palette — Wong (2011) Nature Methods
# ---------------------------------------------------------------------------

WONG_PALETTE = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#000000",  # black
]

# ---------------------------------------------------------------------------
# Font size presets
# ---------------------------------------------------------------------------

_FONT_PRESETS: dict[str, tuple[int, int, int]] = {
    #                  small  medium  large
    "publication":    (6,     8,      9),
    "screen":         (9,     11,     13),
    "presentation":   (12,    14,     16),
}
_DEFAULT_FONTSIZE = "screen"

# ---------------------------------------------------------------------------
# Figure size presets  (width_inches, height_inches)
# ---------------------------------------------------------------------------

_GOLDEN = (1 + math.sqrt(5)) / 2  # ≈ 1.618


def _auto_height(w: float) -> float:
    return round(w / _GOLDEN, 2)


_SIZE_PRESETS: dict[str, tuple[float, float]] = {
    "single": (3.50, _auto_height(3.50)),
    "double": (7.20, _auto_height(7.20)),
    "full":   (7.20, _auto_height(7.20)),
}
_DEFAULT_SIZE = "5x3.5"


def parse_figsize(size_str: str) -> tuple[float, float]:
    """Parse --size argument into (width, height) inches."""
    if size_str in _SIZE_PRESETS:
        return _SIZE_PRESETS[size_str]
    try:
        w_str, h_str = size_str.lower().split("x")
        return float(w_str), float(h_str)
    except (ValueError, AttributeError):
        logger.warning("Cannot parse --size %r; using default 5x3.5", size_str)
        return 5.0, 3.5


# ---------------------------------------------------------------------------
# rcParams application
# ---------------------------------------------------------------------------

def apply_style(args: argparse.Namespace) -> None:
    """
    Apply font sizes, palette, and optional user stylesheets to rcParams.
    Call once before creating the figure.
    """
    import matplotlib.pyplot as plt
    preset = getattr(args, "fontsize", _DEFAULT_FONTSIZE)
    if preset not in _FONT_PRESETS:
        logger.warning("Unknown --fontsize %r; using 'screen'", preset)
        preset = "screen"

    small, medium, large = _FONT_PRESETS[preset]

    matplotlib.rcParams.update({
        "font.family":            "sans-serif",
        "font.sans-serif":        ["Arial", "Helvetica", "DejaVu Sans"],
        "xtick.labelsize":        small,
        "ytick.labelsize":        small,
        "axes.labelsize":         medium,
        "legend.fontsize":        medium,
        "axes.titlesize":         large,
        "legend.title_fontsize":  large,
        "figure.dpi":             100,
        "savefig.dpi":            300,
        "axes.prop_cycle":        matplotlib.cycler(color=WONG_PALETTE),
    })

    # Optional user stylesheets (applied after our base to allow overrides)
    styles = getattr(args, "styles", None) or []
    valid = [s for s in styles if s in plt.style.available]
    if valid:
        plt.style.use(valid)
        # Re-enforce tick visibility which some styles suppress
        matplotlib.rcParams["xtick.bottom"] = True
        matplotlib.rcParams["ytick.left"] = True


# ---------------------------------------------------------------------------
# Figure / axes creation
# ---------------------------------------------------------------------------

def make_figure(
    args: argparse.Namespace,
    nrows: int = 1,
    ncols: int = 1,
) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    """
    Create a figure with nrows × ncols axes.

    Returns (fig, axes) where axes is always a 2-D numpy array of shape
    (nrows, ncols) for uniform indexing even when nrows=ncols=1.
    """
    import matplotlib.pyplot as plt
    size_str = getattr(args, "size", _DEFAULT_SIZE)
    w, h = parse_figsize(size_str)

    # Scale height proportionally when using a subplot grid
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(w * ncols, h * nrows),
        squeeze=False,
    )
    return fig, axes


# ---------------------------------------------------------------------------
# Subplot grid for --subgraphcol
# ---------------------------------------------------------------------------

def subgraph_layout(n_groups: int, ncols_hint: Optional[int] = None) -> tuple[int, int]:
    """Return (nrows, ncols) for a grid of n_groups subplots."""
    if ncols_hint is not None:
        ncols = max(1, ncols_hint)
    else:
        ncols = min(n_groups, 3)  # default: at most 3 columns
    nrows = math.ceil(n_groups / ncols)
    return nrows, ncols


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _to_float(s: str, fallback: float) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return fallback


def apply_limits(ax: matplotlib.axes.Axes, args: argparse.Namespace) -> None:
    """Apply --xlim / --ylim / --logx / --logy / --xmargin / --ymargin."""
    xlim = getattr(args, "xlim", None)
    ylim = getattr(args, "ylim", None)
    logx = getattr(args, "logx", False)
    logy = getattr(args, "logy", False)
    xmargin = getattr(args, "xmargin", False)
    ymargin = getattr(args, "ymargin", False)

    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")

    if xmargin:
        xmin, xmax = ax.get_xlim()
        r = xmax - xmin
        ax.set_xlim(xmin - 0.01 * r, xmax + 0.01 * r)
    if ymargin:
        ymin, ymax = ax.get_ylim()
        r = ymax - ymin
        ax.set_ylim(ymin - 0.01 * r, ymax + 0.01 * r)

    if xlim is not None:
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(
            _to_float(xlim[0], xmin),
            _to_float(xlim[1], xmax),
        )
    if ylim is not None:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(
            _to_float(ylim[0], ymin),
            _to_float(ylim[1], ymax),
        )


def apply_ticks(ax: matplotlib.axes.Axes, args: argparse.Namespace) -> None:
    """Apply --xticks / --yticks major/minor locators."""
    xticks = getattr(args, "xticks", None)
    yticks = getattr(args, "yticks", None)
    if xticks is not None:
        ax.xaxis.set_major_locator(ticker.MultipleLocator(xticks[0]))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(xticks[1]))
    if yticks is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(yticks[0]))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(yticks[1]))


def apply_labels(ax: matplotlib.axes.Axes, args: argparse.Namespace,
                 default_xlabel: str = "", default_ylabel: str = "") -> None:
    """Apply title, xlabel, ylabel, defaulting to column names."""
    xlabel = getattr(args, "xlabel", None) or default_xlabel
    ylabel = getattr(args, "ylabel", None) or default_ylabel
    title  = getattr(args, "title", "") or ""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ---------------------------------------------------------------------------
# Group label
# ---------------------------------------------------------------------------

def make_grouplabel(groupname, groupcols: list[str] | None = None) -> str:
    """Format a groupby key as a readable legend label."""
    if isinstance(groupname, tuple):
        if groupcols and len(groupcols) == len(groupname):
            return "  ".join(f"{k}={v}" for k, v in zip(groupcols, groupname))
        return "-".join(str(x) for x in groupname)
    return str(groupname)


# ---------------------------------------------------------------------------
# Save / show
# ---------------------------------------------------------------------------

def save_or_show(fig: matplotlib.figure.Figure, args: argparse.Namespace) -> None:
    """Save figure to --file (300 dpi) or display interactively."""
    import matplotlib.pyplot as plt
    outfile = getattr(args, "file", None)
    if outfile:
        fig.savefig(outfile, bbox_inches="tight", dpi=300)
        logger.info("Saved figure to %s", outfile)
    else:
        backend = matplotlib.get_backend().lower()
        if backend in ("agg", "cairo", "pdf", "ps", "svg", "pgf"):
            import sys
            print(
                f"Warning: no interactive display (backend: {backend}). "
                "Use -f/--file to save the figure to a file.",
                file=sys.stderr,
            )
        else:
            plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared argparse argument groups
# ---------------------------------------------------------------------------

def add_xy_arguments(parser: argparse.ArgumentParser) -> None:
    """Add x/y column, label, limit, log-scale, margin, tick args."""
    g = parser.add_argument_group("axes")
    g.add_argument("-x", "--xcol",   metavar="COL", help="X-axis column.")
    g.add_argument("-y", "--ycol",   metavar="COL", help="Y-axis column.")
    g.add_argument("-xl", "--xlabel", metavar="TEXT", default=None,
                   help="X-axis label (default: column name).")
    g.add_argument("-yl", "--ylabel", metavar="TEXT", default=None,
                   help="Y-axis label (default: column name).")
    g.add_argument("-xm", "--xlim", nargs=2, metavar=("LO", "HI"), default=None,
                   help="X-axis limits.")
    g.add_argument("-ym", "--ylim", nargs=2, metavar=("LO", "HI"), default=None,
                   help="Y-axis limits.")
    g.add_argument("-lx", "--logx", action="store_true", help="Logarithmic x axis.")
    g.add_argument("-ly", "--logy", action="store_true", help="Logarithmic y axis.")
    g.add_argument("--xmargin", action="store_true", help="Add 1 %% margin to x limits.")
    g.add_argument("--ymargin", action="store_true", help="Add 1 %% margin to y limits.")
    g.add_argument("-xt", "--xticks", nargs=2, type=float, metavar=("MAJOR", "MINOR"),
                   default=None, help="X-axis major and minor tick spacing.")
    g.add_argument("-yt", "--yticks", nargs=2, type=float, metavar=("MAJOR", "MINOR"),
                   default=None, help="Y-axis major and minor tick spacing.")


def add_figure_arguments(parser: argparse.ArgumentParser) -> None:
    """Add figure-level args: title, size, fontsize, styles, file."""
    g = parser.add_argument_group("figure")
    g.add_argument("-t", "--title", default="", metavar="TEXT",
                   help="Figure title.")
    g.add_argument("--size", default=_DEFAULT_SIZE, metavar="PRESET|WxH",
                   help="Figure size: single (3.5\"), double (7.2\"), full, or WxH "
                        f"in inches (default: {_DEFAULT_SIZE}).")
    g.add_argument("--fontsize",
                   choices=list(_FONT_PRESETS),
                   default=_DEFAULT_FONTSIZE,
                   help="Font size preset (default: screen). "
                        "small=tick labels, medium=axis labels/legend, "
                        "large=title/legend title.")
    g.add_argument("--styles", nargs="*", metavar="STYLE", default=None,
                   help="Matplotlib stylesheets to apply (e.g. seaborn-v0_8).")
    g.add_argument("-f", "--file", default=None, metavar="PATH",
                   help="Save figure to file instead of displaying it. "
                        "Format inferred from extension (.png, .pdf, .svg, …).")
    g.add_argument("--usetex", action="store_true",
                   help="Render text with LaTeX (requires a TeX installation).")


def add_group_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --groupcol and --subgraphcol."""
    g = parser.add_argument_group("grouping")
    g.add_argument("-g", "--groupcol", nargs="+", default=None, metavar="COL",
                   help="Colour-code by this column (one series per value).")
    g.add_argument("--subgraphcol", nargs="+", default=None, metavar="COL",
                   help="Split into a subplot grid by this column.")
    g.add_argument("--ncols", type=int, default=None, metavar="N",
                   help="Number of columns in the subplot grid "
                        "(default: min(groups, 3)).")


def add_legend_arguments(parser: argparse.ArgumentParser) -> None:
    """Add --legend / --legendtitle / --legendloc."""
    g = parser.add_argument_group("legend")
    g.add_argument("--legend", default=None, metavar="TEXT",
                   help="Legend label (when not using --groupcol).")
    g.add_argument("--legendtitle", default=None, metavar="TEXT",
                   help="Legend title.")
    g.add_argument("--legendloc", nargs=2, type=float, metavar=("X", "Y"),
                   default=None,
                   help="Legend position as (x, y) in axes coordinates.")
