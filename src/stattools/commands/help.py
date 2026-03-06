"""
stattools.commands.help — the ``dfstat help`` subcommand.

  dfstat help              — list all available subcommands
  dfstat help <command>    — show full argparse help for one subcommand
"""

import argparse

from stattools.commands.base import BaseCommand


class HelpCommand(BaseCommand):
    """Show help for dfstat subcommands."""

    name = "help"
    help = "Show detailed help for a subcommand"

    def __init__(self, available_commands: list[BaseCommand] | None = None) -> None:
        # Receives the live command list from commands/__init__.py so it can
        # introspect any peer command without circular imports.
        self.available_commands: list[BaseCommand] = available_commands or []

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "command_name",
            nargs="?",
            metavar="COMMAND",
            help="Name of the subcommand to get help for",
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.command_name is None:
            # List all commands
            print("Available commands:")
            for cmd in self.available_commands:
                print(f"  {cmd.name:<16} {cmd.help}")
            print("\nUse 'dfstat help <command>' for detailed help on a command.")
            return

        cmd = next(
            (c for c in self.available_commands if c.name == args.command_name),
            None,
        )
        if cmd is None:
            raise ValueError(f"Unknown command: {args.command_name!r}")

        # Build a temporary parser for this command so argparse formats the
        # help text exactly as it would appear during normal use.
        temp_parser = argparse.ArgumentParser(
            prog=f"dfstat {cmd.name}",
            description=cmd.help,
        )
        cmd.add_arguments(temp_parser)
        temp_parser.print_help()
