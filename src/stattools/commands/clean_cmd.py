"""
stattools.commands.clean_cmd — the ``dfstat clean`` subcommand.

Removes all temp parquet pipe files left in DFSTAT_TMPDIR.  Useful after a
pipeline is interrupted before the final ``dfstat print`` stage has had a
chance to consume (and therefore delete) them.

    dfstat clean
"""

import os
import sys
import argparse

from stattools.commands.base import BaseCommand
from stattools.common.io import DFSTAT_TMPDIR


class CleanCommand(BaseCommand):
    """Remove leftover temp parquet pipe files."""

    name = "clean"
    help = f"Remove temp parquet pipe files from the dfstat temp directory"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # No arguments needed — always cleans DFSTAT_TMPDIR.
        pass

    def execute(self, args: argparse.Namespace) -> None:
        if not os.path.isdir(DFSTAT_TMPDIR):
            print(
                f"Nothing to clean (directory does not exist: {DFSTAT_TMPDIR})",
                file=sys.stderr,
            )
            return

        files = [f for f in os.listdir(DFSTAT_TMPDIR) if f.endswith(".parquet")]
        errors = 0
        for f in files:
            try:
                os.unlink(os.path.join(DFSTAT_TMPDIR, f))
            except OSError as exc:
                print(f"Warning: could not remove {f}: {exc}", file=sys.stderr)
                errors += 1

        removed = len(files) - errors
        print(
            f"Removed {removed} file(s) from {DFSTAT_TMPDIR}"
            + (f" ({errors} error(s))" if errors else ""),
            file=sys.stderr,
        )
