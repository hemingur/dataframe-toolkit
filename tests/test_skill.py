"""
Keep the agent skill in skills/dftk/ in step with the CLI.

The skill is read by coding agents instead of --help, so a renamed command or
option there would teach them a command line that no longer works.  These
tests check every dftk command line in the skill's code against the real
parsers, and that the command overview lists every command.
"""

import argparse
import re
from pathlib import Path

import pytest

from dftk.commands import COMMAND_GROUPS, COMMAND_NAMES, load_command

SKILL_DIR = Path(__file__).parents[1] / "skills" / "dftk"
SKILL_FILES = sorted(SKILL_DIR.rglob("*.md"))

_FENCED = re.compile(r"^```[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)
_INLINE = re.compile(r"`([^`\n]+)`")
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_OPTION = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_COMMAND = re.compile(r"\bdftk\s+([a-z]+)\b(.*)")


def _command_lines(text: str) -> list[tuple[str, str]]:
    """Return (command, arguments) for each dftk invocation in *text*'s code.

    Fenced blocks are joined across backslash continuations and split into
    pipeline stages.  Inline code counts when it starts with a command name,
    with or without the ``dftk`` prefix (the overview tables omit it).
    """
    found = []
    for block in _FENCED.findall(text):
        for line in block.replace("\\\n", " ").splitlines():
            for stage in line.split("|"):
                if m := _COMMAND.search(stage):
                    found.append((m.group(1), m.group(2)))
    prose = _FENCED.sub("", text)
    for span in _INLINE.findall(prose):
        for stage in re.split(r"\||\s/\s", span):
            stage = stage.strip()
            prefixed = stage.startswith("dftk ")
            words = stage.removeprefix("dftk ").split(maxsplit=1)
            if not words or words[0].startswith("<"):  # `dftk <command> --help`
                continue
            if prefixed or words[0] in COMMAND_NAMES:
                found.append((words[0], words[1] if len(words) > 1 else ""))
    return found


def _option_strings(name: str) -> set[str]:
    parser = argparse.ArgumentParser(prog=f"dftk {name}")
    load_command(name).add_arguments(parser)
    return {opt for action in parser._actions for opt in action.option_strings}


ALL_LINES = [
    (path.relative_to(SKILL_DIR).as_posix(), cmd, rest)
    for path in SKILL_FILES
    for cmd, rest in _command_lines(path.read_text())
]


class TestSkill:
    def test_skill_files_exist(self):
        assert (SKILL_DIR / "SKILL.md").is_file()
        assert ALL_LINES, "no dftk command lines found; the parser here is broken"

    def test_frontmatter_names_the_skill(self):
        text = (SKILL_DIR / "SKILL.md").read_text()
        assert text.startswith("---\nname: dftk\ndescription: ")

    @pytest.mark.parametrize("path,cmd,rest", ALL_LINES)
    def test_command_line_uses_real_command_and_options(self, path, cmd, rest):
        assert cmd in COMMAND_NAMES, f"{path}: unknown command {cmd!r}"
        options = set(_OPTION.findall(_QUOTED.sub("", rest)))
        unknown = options - _option_strings(cmd)
        assert not unknown, f"{path}: dftk {cmd} has no option(s) {sorted(unknown)}"

    def test_overview_lists_every_command(self):
        text = (SKILL_DIR / "SKILL.md").read_text()
        grouped = [name for _, entries in COMMAND_GROUPS for name, _ in entries]
        missing = [
            name
            for name in grouped
            if f"`{name}" not in text and f"`dftk {name}" not in text
        ]
        assert not missing, f"SKILL.md does not mention: {missing}"
