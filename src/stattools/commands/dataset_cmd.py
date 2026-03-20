"""
stattools.commands.dataset_cmd — the ``dfstat dataset`` subcommand.

Provides access to curated example datasets from seaborn, statsmodels,
and (optionally) pydataset.  Datasets are written to stdout as TSV or
into the dfstat parquet pipe with -o.

Examples
--------
    dfstat dataset iris
    dfstat dataset iris --source seaborn
    dfstat dataset longley --source statsmodels
    dfstat dataset --list
    dfstat dataset --list --source seaborn
    dfstat dataset iris -o | dfstat stat -c sepal_length -g species
"""

import argparse
import sys

import pandas as pd

from stattools.commands.base import BaseCommand
from stattools.common.io import io


# ---------------------------------------------------------------------------
# Per-source load/list helpers
# ---------------------------------------------------------------------------

def _load_seaborn(name: str) -> pd.DataFrame:
    import seaborn as sns
    return sns.load_dataset(name)


def _list_seaborn() -> list[tuple[str, str, str]]:
    import seaborn as sns
    return [("seaborn", n, "") for n in sorted(sns.get_dataset_names())]


def _load_statsmodels(name: str) -> pd.DataFrame:
    import statsmodels.api as sm
    mod = getattr(sm.datasets, name, None)
    if mod is not None and hasattr(mod, "load_pandas"):
        return mod.load_pandas().data
    raise ValueError(f"statsmodels dataset {name!r} not found")


def _list_statsmodels() -> list[tuple[str, str, str]]:
    import statsmodels.api as sm
    rows = []
    for name in sorted(dir(sm.datasets)):
        if name.startswith("_"):
            continue
        mod = getattr(sm.datasets, name, None)
        if mod is None or not hasattr(mod, "load_pandas"):
            continue
        title = (getattr(mod, "TITLE", "") or "").strip()
        desc = title or (getattr(mod, "DESCRSHORT", "") or "").strip()
        rows.append(("statsmodels", name, desc))
    return rows


try:
    import pydataset as _pydataset  # noqa: F401
    _HAS_PYDATASET = True
except ImportError:
    _HAS_PYDATASET = False


def _load_pydataset(name: str) -> pd.DataFrame:
    import pydataset
    return pydataset.data(name)


def _list_pydataset() -> list[tuple[str, str, str]]:
    import pydataset
    index = pydataset.data()
    return [
        ("pydataset", str(row["item"]), str(row.get("title", "")))
        for _, row in index.iterrows()
    ]


# Registry — built at import time so choices are accurate
_SOURCES: dict[str, tuple] = {
    "seaborn": (_load_seaborn, _list_seaborn),
    "statsmodels": (_load_statsmodels, _list_statsmodels),
}
if _HAS_PYDATASET:
    _SOURCES["pydataset"] = (_load_pydataset, _list_pydataset)


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------

class DatasetCommand(BaseCommand):
    name = "dataset"
    help = "Load a curated example dataset (seaborn, statsmodels, pydataset)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        io.parser_output(parser)

        g = parser.add_argument_group("dataset options")
        g.add_argument(
            "NAME",
            nargs="?",
            default=None,
            help=(
                "Dataset name (e.g. 'iris', 'longley').  "
                "Omit with --list to show all available datasets."
            ),
        )
        g.add_argument(
            "--source",
            nargs="+",
            default=None,
            choices=list(_SOURCES),
            metavar="SOURCE",
            help=(
                f"Source(s) to search: {', '.join(_SOURCES)}.  "
                "Default: try all sources in order."
            ),
        )
        g.add_argument(
            "--list",
            action="store_true",
            dest="list_datasets",
            help="List available datasets as TSV (source, name, description).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        sources = args.source or list(_SOURCES)

        if args.list_datasets:
            self._print_list(sources)
            return

        if args.NAME is None:
            raise ValueError(
                "Provide a dataset NAME, or use --list to browse available datasets."
            )

        df = self._load(args.NAME, sources)
        io.printdf(df, args)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _load(self, name: str, sources: list[str]) -> pd.DataFrame:
        errors: list[str] = []
        for src in sources:
            load_fn, _ = _SOURCES[src]
            try:
                return load_fn(name)
            except Exception as exc:
                errors.append(f"  {src}: {exc}")
        raise ValueError(
            f"Dataset {name!r} not found in: {', '.join(sources)}.\n"
            + "\n".join(errors)
        )

    def _print_list(self, sources: list[str]) -> None:
        print("source\tname\tdescription")
        for src in sources:
            _, list_fn = _SOURCES[src]
            try:
                for source, name, desc in list_fn():
                    print(f"{source}\t{name}\t{desc}")
            except Exception as exc:
                print(f"Warning: could not list {src} datasets: {exc}", file=sys.stderr)
