"""
dftk.cli — entry point for the ``dftk`` command.

All subcommands are discovered through the registry in
dftk/commands/__init__.py.  This file contains no command-specific
logic and never needs to change when a new subcommand is added.

Only the command being run is imported, so e.g. ``dftk print`` does not pay
for importing scipy, statsmodels or matplotlib.  The full set is imported only
when it is needed: ``dftk --help``, no command, or an unknown command name.
"""

import argparse
import logging
import os
import signal
import sys

from dftk import __version__
from dftk.commands import COMMAND_NAMES, load_command, load_command_groups
from dftk.commands.base import BaseCommand
from dftk.common.io import DFTK_TMPDIR

logger = logging.getLogger("dftk")
logging.basicConfig(format="%(asctime)s %(module)s %(message)s", level=logging.INFO)


def _commands_for(argv: list[str]) -> list[BaseCommand]:
    """Return the commands whose subparsers *argv* needs.

    The top-level parser has no options that take a value, so a valid
    invocation always has the command name first.
    """
    if argv and argv[0] in COMMAND_NAMES:
        return [load_command(argv[0])]
    # `dftk --help`, no command, or an unknown name: load everything so the
    # command listing and argparse's "invalid choice" message are complete.
    grouped = [cmd for _, cmds in load_command_groups() for cmd in cmds]
    return grouped + [load_command("help")]


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

    for cmd in _commands_for(sys.argv[1:]):
        cmd_parser = subparsers.add_parser(cmd.name, help=cmd.help)
        cmd.add_arguments(cmd_parser)
        cmd_parser.set_defaults(command_instance=cmd)

    args = parser.parse_args()

    try:
        args.command_instance.execute(args)
        sys.stdout.flush()
    except BrokenPipeError:
        # The reader closed the pipe early (e.g. `dftk ... | head`).  Exit
        # quietly with the SIGPIPE status like cat/grep, and point stdout at
        # devnull so the interpreter's final flush doesn't raise again.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(128 + signal.SIGPIPE)
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
