"""
Help texts: an example split over lines with a trailing backslash must keep
its line breaks.  In a normal (not raw) Python string a lone backslash at a
line end joins the lines, and the example prints as one long line.
"""

import argparse
import re

import pytest

from dftk.commands import COMMAND_GROUPS, load_command

# text, then 3+ spaces, then an option or a pipe: two example lines run together
_JOINED = re.compile(r"\S {3,}(\| dftk|--?[A-Za-z])")

NAMES = [name for _, entries in COMMAND_GROUPS for name, _ in entries]


@pytest.mark.parametrize("name", NAMES)
def test_examples_keep_their_line_breaks(name):
    parser = argparse.ArgumentParser()
    load_command(name).add_arguments(parser)
    text = (parser.description or "") + "\n" + (parser.epilog or "")
    joined = [line for line in text.splitlines() if _JOINED.search(line.strip())]
    assert not joined, f"dftk {name} -h: lines run together: {joined}"
