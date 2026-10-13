"""
Tests for the command registry in dftk.commands.
"""

from pathlib import Path

import pytest

import dftk.commands
from dftk.commands import (
    _REGISTRY,
    COMMAND_GROUPS,
    COMMAND_NAMES,
    load_command,
    load_command_groups,
)
from dftk.commands.base import BaseCommand


class TestRegistry:
    @pytest.mark.parametrize("name", COMMAND_NAMES)
    def test_entry_loads_command_with_matching_name(self, name):
        cmd = load_command(name)
        assert isinstance(cmd, BaseCommand)
        assert cmd.name == name

    def test_every_cmd_module_is_registered(self):
        """A new <name>_cmd.py that was never added to COMMAND_GROUPS is invisible."""
        package_dir = Path(dftk.commands.__file__).parent
        modules = {p.stem for p in package_dir.glob("*_cmd.py")}
        registered = {path.split(":")[0] for path in _REGISTRY.values()}
        assert modules - registered == set()

    def test_names_are_unique(self):
        names = [name for _, entries in COMMAND_GROUPS for name, _ in entries]
        assert len(names) == len(set(names))

    def test_help_is_registered_but_not_grouped(self):
        grouped = {name for _, entries in COMMAND_GROUPS for name, _ in entries}
        assert "help" in COMMAND_NAMES
        assert "help" not in grouped

    def test_load_command_groups_keeps_order(self):
        groups = load_command_groups()
        assert [header for header, _ in groups] == [h for h, _ in COMMAND_GROUPS]
        assert [c.name for c in groups[0][1]] == [n for n, _ in COMMAND_GROUPS[0][1]]
