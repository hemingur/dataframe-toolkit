"""
dftk.commands.completion_cmd — dftk completion subcommand.

Writes a Tab-completion script for bash or zsh, generated with shtab from the
real dftk parser: commands, options and their allowed values, and file names
where a command takes a file.  The script is plain shell, so pressing Tab
never starts Python; it is regenerated after an upgrade.
"""

import argparse
import os
from pathlib import Path

from dftk.commands.base import BaseCommand

SHELLS = ("bash", "zsh")

_EPILOG = """\
Tab completes commands, options, the values an option allows (--format,
--fit, --catwidth, ...) and file names. Pressing Tab never starts Python:
the script lists everything, so it is instant.

  dftk completion install           install for your shell ($SHELL)
  dftk completion install --shell zsh
  dftk completion print > FILE      write the script yourself

bash: installs to ~/.local/share/bash-completion/completions/dftk, which
bash-completion loads the first time you press Tab after "dftk" (Ubuntu
and most Linux systems load bash-completion in ~/.bashrc; on macOS install
it with Homebrew). Open a new shell to start using it.

zsh: installs to ~/.zfunc/_dftk. Add this to ~/.zshrc, before compinit or,
with Oh My Zsh, before the line that sources oh-my-zsh.sh:

  fpath+=~/.zfunc

then open a new shell (if it does not complete, rm ~/.zcompdump* once).

Run "dftk completion install" again after upgrading dftk, so new commands
and options complete; "dftk skill install" also refreshes an installed
completion script.
"""


def default_shell() -> str:
    """The user's shell from $SHELL, if dftk supports it, else bash."""
    shell = Path(os.environ.get("SHELL", "")).name
    return shell if shell in SHELLS else "bash"


def install_path(shell: str) -> Path:
    """Where `dftk completion install` puts the script for *shell*."""
    home = Path.home()
    if shell == "zsh":
        return home / ".zfunc" / "_dftk"
    user_dir = os.environ.get("BASH_COMPLETION_USER_DIR")
    if user_dir:
        return Path(user_dir) / "completions" / "dftk"
    data = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(data) / "bash-completion" / "completions" / "dftk"


def _subparsers(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _mark_completions(parser: argparse.ArgumentParser) -> None:
    """Tell shtab which arguments are files or directories, and complete
    `dftk help COMMAND` with the command names."""
    import shtab

    commands = _subparsers(parser)
    for name, sub in commands.items():
        for action in sub._actions:
            if action.nargs == 0 or action.choices:
                continue
            metavar = str(action.metavar or "")
            if name == "help" and not action.option_strings:
                action.choices = sorted(commands)
            elif not action.option_strings and action.dest in (
                "DATAFILE",
                "DATAFILES",
                "PARQUET",
            ):
                action.complete = shtab.FILE
            elif metavar in ("FILE", "PATH"):
                action.complete = shtab.FILE
            elif metavar == "DIR":
                action.complete = shtab.DIRECTORY


def completion_script(shell: str) -> str:
    """The completion script for *shell*, from every dftk command's parser."""
    import shtab

    from dftk.cli import all_commands, build_parser

    parser = build_parser(all_commands())
    _mark_completions(parser)
    return shtab.complete(parser, shell=shell)


def install(shell: str) -> Path:
    path = install_path(shell)
    script = completion_script(shell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    return path


def refresh_installed() -> list[Path]:
    """Rewrite the completion scripts already installed (after an upgrade)."""
    return [install(shell) for shell in SHELLS if install_path(shell).is_file()]


class CompletionCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "completion"

    @property
    def help(self) -> str:
        return "Tab completion for bash and zsh"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG
        parser.add_argument(
            "action",
            choices=["install", "print"],
            help="install: write the script where the shell finds it; "
            "print: write it to stdout.",
        )
        parser.add_argument(
            "--shell",
            choices=SHELLS,
            default=None,
            help="Shell to complete in (default: from $SHELL, else bash).",
        )

    def execute(self, args: argparse.Namespace) -> None:
        shell = args.shell or default_shell()
        if args.action == "print":
            print(completion_script(shell), end="")
            return
        path = install(shell)
        print(f"Installed {shell} completion for dftk in {path}")
        if shell == "zsh":
            print(
                "Add 'fpath+=~/.zfunc' to ~/.zshrc before compinit (with Oh My "
                "Zsh: before it sources oh-my-zsh.sh), then open a new shell."
            )
        else:
            print("Open a new shell to use it.")
        print('Run "dftk completion install" again after upgrading dftk.')
