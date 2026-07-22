"""
dftk.commands.info_cmd — dftk info subcommand.

Lightweight structural overview of a DataFrame: per-column dtype, null
counts, and memory usage. Complements `describe`, which profiles
statistical distributions and quality flags instead.

Example
-------
    dftk info data.tsv
    dftk info data.tsv --summary
"""

import argparse
import sys

import pandas as pd

from dftk.commands.base import BaseCommand
from dftk.common.io import io


def _info(df: pd.DataFrame) -> pd.DataFrame:
    mem = df.memory_usage(deep=True, index=False)
    return pd.DataFrame(
        [
            dict(
                name=col,
                dtype=str(df[col].dtype),
                non_null=int(df[col].notna().sum()),
                null=int(df[col].isna().sum()),
                memory_bytes=int(mem[col]),
            )
            for col in df.columns
        ]
    )


def _summary_line(df: pd.DataFrame) -> str:
    n_rows, n_cols = df.shape
    total_mem = int(df.memory_usage(deep=True).sum())
    n_dupes = int(df.duplicated().sum())
    return (
        f"Dataset: {n_rows} rows x {n_cols} columns, "
        f"{total_mem} bytes, {n_dupes} duplicated row(s)"
    )


class InfoCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "info"

    @property
    def help(self) -> str:
        return "Structural overview: per-column dtype, null counts, memory usage"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        self.add_io_arguments(parser)

        g = parser.add_argument_group("info options")
        g.add_argument(
            "--summary",
            action="store_true",
            help="Print dataset totals (shape, memory, duplicate rows) to stderr.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        df = io.read(args)

        if getattr(args, "summary", False):
            print(_summary_line(df), file=sys.stderr)

        io.printdf(_info(df), args)
