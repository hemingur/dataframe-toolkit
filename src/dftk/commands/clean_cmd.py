"""
dftk.commands.clean_cmd — the ``dftk clean`` subcommand.

Removes all temp parquet pipe files left in DFTK_TMPDIR.  Useful after a
pipeline is interrupted before the final ``dftk print`` stage has had a
chance to consume (and therefore delete) them.

    dftk clean
"""

import argparse
import os
import sys

from dftk.commands.base import BaseCommand
from dftk.common.io import DFTK_TMPDIR


class CleanCommand(BaseCommand):
    """Remove leftover temp parquet pipe files."""

    name = "clean"
    help = "Remove temp parquet pipe files from the dftk temp directory"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # No arguments needed — always cleans DFTK_TMPDIR.
        pass

    def execute(self, args: argparse.Namespace) -> None:
        if not os.path.isdir(DFTK_TMPDIR):
            print(
                f"Nothing to clean (directory does not exist: {DFTK_TMPDIR})",
                file=sys.stderr,
            )
            return

        files = [f for f in os.listdir(DFTK_TMPDIR) if f.endswith(".parquet")]
        errors = 0
        for f in files:
            try:
                os.unlink(os.path.join(DFTK_TMPDIR, f))
            except OSError as exc:
                print(f"Warning: could not remove {f}: {exc}", file=sys.stderr)
                errors += 1

        removed = len(files) - errors
        print(
            f"Removed {removed} file(s) from {DFTK_TMPDIR}"
            + (f" ({errors} error(s))" if errors else ""),
            file=sys.stderr,
        )
