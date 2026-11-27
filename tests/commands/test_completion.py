"""Tests for dftk.commands.completion_cmd (dftk completion install / print)."""

import argparse
import shutil
import subprocess

import pytest

from dftk.cli import all_commands, build_parser
from dftk.commands.completion_cmd import (
    CompletionCommand,
    completion_script,
    default_shell,
    install_path,
    refresh_installed,
)
from tests.conftest import make_args


@pytest.fixture(scope="module")
def bash_script():
    return completion_script("bash")


@pytest.fixture(scope="module")
def zsh_script():
    return completion_script("zsh")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.delenv("BASH_COMPLETION_USER_DIR", raising=False)
    return tmp_path


def _complete(script_path, cwd, *words):
    """Run the bash completion function for `dftk WORDS...` (the last word is
    the one being completed) and return the candidates."""
    driver = r"""
source "$1"; shift
COMP_WORDS=("$@"); COMP_CWORD=$(( ${#COMP_WORDS[@]} - 1 ))
COMP_LINE="${COMP_WORDS[*]}"; COMP_POINT=${#COMP_LINE}
fn=$(complete -p dftk | sed -E 's/.*-F ([^ ]+).*/\1/')
$fn dftk "${COMP_WORDS[COMP_CWORD]}" "${COMP_WORDS[COMP_CWORD-1]}"
printf '%s\n' "${COMPREPLY[@]}"
"""
    out = subprocess.run(
        ["bash", "-c", driver, "driver", str(script_path), "dftk", *words],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sorted(w for w in out.splitlines() if w)


@pytest.fixture(scope="module")
def bash_file(tmp_path_factory, bash_script):
    path = tmp_path_factory.mktemp("completion") / "dftk.bash"
    path.write_text(bash_script)
    return path


@pytest.fixture
def files(tmp_path):
    for name in ("data.tsv", "data.parquet", "other.txt"):
        (tmp_path / name).write_text("")
    (tmp_path / "datadir").mkdir()
    return tmp_path


def _all_options():
    parser = build_parser(all_commands())
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {
        name: [o for a in sp._actions for o in a.option_strings]
        for name, sp in sub.choices.items()
    }


needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="no bash")


class TestScript:
    def test_lists_every_command_and_option(self, bash_script, zsh_script):
        for command, options in _all_options().items():
            for script in (bash_script, zsh_script):
                assert command in script
            missing = [o for o in options if o not in bash_script]
            assert not missing, f"{command}: {missing}"

    @needs_bash
    def test_bash_syntax(self, bash_file):
        subprocess.run(["bash", "-n", str(bash_file)], check=True)

    @pytest.mark.skipif(shutil.which("zsh") is None, reason="no zsh")
    def test_zsh_syntax(self, tmp_path, zsh_script):
        path = tmp_path / "_dftk"
        path.write_text(zsh_script)
        subprocess.run(["zsh", "-n", str(path)], check=True)

    def test_zsh_is_a_compdef(self, zsh_script):
        assert zsh_script.startswith("#compdef dftk")


@needs_bash
class TestBashCompletes:
    @pytest.mark.parametrize(
        "words, expected",
        [
            (["sc"], ["scale", "scat"]),
            (["scat", "--ab"], ["--abline"]),
            (["scat", "--fit", ""], ["lowess", "ols"]),
            (["print", "--format", ""], ["json", "jsonl", "md", "tex", "tsv"]),
            (["completion", ""], ["install", "print"]),
            (["help", "sc"], ["scale", "scat"]),
        ],
    )
    def test_commands_options_and_values(self, bash_file, files, words, expected):
        assert _complete(bash_file, files, *words) == expected

    @pytest.mark.parametrize(
        "words",
        [
            ["scat", "da"],  # DATAFILE
            ["print", "data.tsv", "-o", "da"],  # -o FILE
            ["merge", "-l", "da"],
            ["concat", "data.tsv", "da"],  # DATAFILES
            ["annotate", "da"],  # PARQUET
            ["scat", "-x", "a", "-y", "b", "-f", "da"],  # -f PATH
        ],
    )
    def test_file_names(self, bash_file, files, words):
        assert _complete(bash_file, files, *words) == [
            "data.parquet",
            "data.tsv",
            "datadir",
        ]

    def test_directories(self, bash_file, files):
        assert _complete(bash_file, files, "skill", "install", "--dir", "da") == [
            "datadir"
        ]


class TestInstall:
    def test_paths(self, home, monkeypatch):
        bash = home / ".local" / "share" / "bash-completion" / "completions" / "dftk"
        assert install_path("bash") == bash
        assert install_path("zsh") == home / ".zfunc" / "_dftk"
        monkeypatch.setenv("XDG_DATA_HOME", str(home / "xdg"))
        assert install_path("bash") == home / "xdg/bash-completion/completions/dftk"
        monkeypatch.setenv("BASH_COMPLETION_USER_DIR", str(home / "bc"))
        assert install_path("bash") == home / "bc" / "completions" / "dftk"

    @pytest.mark.parametrize(
        "shell, expected",
        [
            ("/bin/zsh", "zsh"),
            ("/usr/bin/bash", "bash"),
            ("/usr/bin/fish", "bash"),
            ("", "bash"),
        ],
    )
    def test_default_shell(self, monkeypatch, shell, expected):
        monkeypatch.setenv("SHELL", shell)
        assert default_shell() == expected

    @pytest.mark.parametrize("shell", ["bash", "zsh"])
    def test_install_writes_the_script(self, home, capsys, shell):
        CompletionCommand().execute(make_args(action="install", shell=shell))
        path = install_path(shell)
        assert path.read_text() == completion_script(shell)
        out = capsys.readouterr().out
        assert f"Installed {shell} completion for dftk in {path}" in out
        assert ("fpath+=~/.zfunc" in out) == (shell == "zsh")

    def test_print(self, capsys, bash_script):
        CompletionCommand().execute(make_args(action="print", shell="bash"))
        assert capsys.readouterr().out == bash_script

    def test_refresh_only_installed(self, home):
        assert refresh_installed() == []
        zsh = install_path("zsh")
        zsh.parent.mkdir(parents=True)
        zsh.write_text("# old\n")
        assert refresh_installed() == [zsh]
        assert zsh.read_text().startswith("#compdef dftk")
        assert not install_path("bash").exists()
