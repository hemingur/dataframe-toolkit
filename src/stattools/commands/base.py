"""
stattools.commands.base — abstract base class for all dfstat subcommands.

Every subcommand must:

  1. Inherit from BaseCommand.
  2. Define the ``name`` and ``help`` class attributes (or properties).
  3. Implement ``add_arguments(parser)`` to declare its CLI arguments.
  4. Implement ``execute(args)`` to run the command.

Most commands also call ``self.add_io_arguments(parser)`` inside
``add_arguments`` to attach the standard dfstat read + output argument groups.

Example skeleton::

    from stattools.commands.base import BaseCommand
    from stattools.common.io import io

    class MyCommand(BaseCommand):
        name = "mycommand"
        help = "Does something useful"

        def add_arguments(self, parser):
            self.add_io_arguments(parser)           # adds DATAFILE, -o, etc.
            g = parser.add_argument_group("mycommand options")
            g.add_argument("-x", "--xcol", ...)

        def execute(self, args):
            df = io.read(args)
            # ... transform df ...
            io.printdf(df, args)
"""

import argparse
from abc import ABC, abstractmethod


class BaseCommand(ABC):
    """Abstract base class for all dfstat subcommands."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Command name as it appears on the CLI (e.g. ``"stat"``)."""

    @property
    @abstractmethod
    def help(self) -> str:
        """One-line description shown in ``dfstat --help``."""

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Declare all CLI arguments for this command on *parser*."""

    @abstractmethod
    def execute(self, args: argparse.Namespace) -> None:
        """Run the command using the parsed *args*."""

    # ------------------------------------------------------------------ #
    # Shared helpers                                                        #
    # ------------------------------------------------------------------ #

    def add_io_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Attach the standard dfstat read + output argument groups to *parser*.

        Adds:
          * All ``io.parser_read`` arguments  (DATAFILE, --backend, --noheader, …)
          * All ``io.parser_output`` arguments (-o/--output, --select, --drop, …)

        Call this at the top of ``add_arguments`` for any command that reads
        tabular data and produces tabular output.
        """
        from stattools.common.io import io
        io.parser_read(parser)
        io.parser_output(parser)

    def add_read_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Attach only the read argument group to *parser* (no output args).

        Use this for commands that read tabular data but produce non-tabular
        output (e.g. plots), so that --select, --drop, --round, etc. are not
        exposed.
        """
        from stattools.common.io import io
        io.parser_read(parser)
