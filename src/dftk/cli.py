"""
dftk.cli — entry point for the ``dftk`` command.

All subcommands are discovered through the COMMANDS list in
dftk/commands/__init__.py.  This file contains no command-specific
logic and never needs to change when a new subcommand is added.
"""

import argparse
import logging
import sys

from dftk import __version__
from dftk.commands import COMMANDS
from dftk.common.io import DFTK_TMPDIR

logger = logging.getLogger("dftk")
logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dftk",
        description="DataFrame analysis and manipulation toolkit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Pipe mode — use -o alone to pass data between commands:\n"
            "  dftk stat -g grp -s val -o \\\n"
            "    | dftk pivot -i grp -v val_mean -o \\\n"
            "    | dftk print\n\n"
            f"Temp pipe files are stored in {DFTK_TMPDIR}\n"
            "Run 'dftk clean' to remove any leftover pipe files."
        ),
    )

    parser.add_argument("--version", action="version", version=f"dftk {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    for cmd in COMMANDS:
        cmd_parser = subparsers.add_parser(cmd.name, help=cmd.help)
        cmd.add_arguments(cmd_parser)
        cmd_parser.set_defaults(command_instance=cmd)

    args = parser.parse_args()

    try:
        args.command_instance.execute(args)
    except FileNotFoundError as exc:
        print(f"File error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Processing error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        import traceback

        print(
            f"Unexpected error ({type(exc).__name__}): {exc}\n"
            "Please report this with the full traceback below.",
            file=sys.stderr,
        )
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
