"""
dftk.commands.help — the ``dftk help`` subcommand.

  dftk help              — list all available subcommands
  dftk help <command>    — show full argparse help for one subcommand
"""

import argparse

from dftk.commands import COMMAND_NAMES, load_command, load_command_groups
from dftk.commands.base import BaseCommand


class HelpCommand(BaseCommand):
    """Show help for dftk subcommands."""

    name = "help"
    help = "Show detailed help for a subcommand"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "command_name",
            nargs="?",
            metavar="COMMAND",
            help="Name of the subcommand to get help for",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.command_name is None:
            for header, cmds in load_command_groups():
                print(f"\n{header}:")
                for cmd in cmds:
                    print(f"  {cmd.name:<16} {cmd.help}")
            print("\nUse 'dftk help <command>' for detailed help on a command.")
            return

        if args.command_name not in COMMAND_NAMES:
            raise ValueError(f"Unknown command: {args.command_name!r}")
        cmd = load_command(args.command_name)

        # Build a temporary parser for this command so argparse formats the
        # help text exactly as it would appear during normal use.
        temp_parser = argparse.ArgumentParser(
            prog=f"dftk {cmd.name}",
            description=cmd.help,
        )
        cmd.add_arguments(temp_parser)
        temp_parser.print_help()
