"""
dftk.commands.skill_cmd — dftk skill subcommand.

Installs the agent skill that ships with dftk (SKILL.md and its references)
into the skills folder of a coding agent, so the agent knows dftk's
conventions without a copy from GitHub.
"""

import argparse
import shutil
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dftk.commands.base import BaseCommand

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "dftk"

# Agents whose skills folder is <home>/<folder>/skills
_AGENTS = {"claude": ".claude", "codex": ".codex"}

_EPILOG = """\
dftk ships an agent skill: a short guide that coding agents such as Claude
Code and Codex load when a task involves dftk (piping conventions, which
command to use for what, Snakemake and Make patterns).

  dftk skill install            install for every agent found: ~/.claude
                                and/or ~/.codex (skills/dftk inside each)
  dftk skill install --codex    only Codex (creates ~/.codex/skills)
  dftk skill install --dir DIR  into DIR/dftk, for another agent
  dftk skill path               print the folder of the packaged skill

Run "dftk skill install" again after upgrading dftk, so the skill matches
the installed version. It replaces an earlier copy of the dftk skill, and
nothing else.
"""


def _dftk_version() -> str:
    try:
        return version("dataframe-toolkit")
    except PackageNotFoundError:
        return "unknown"


def _skills_folders(args: argparse.Namespace) -> list[Path]:
    """Return the skills folders to install into (each gets a dftk/ inside)."""
    if args.dir is not None:
        return [Path(args.dir).expanduser()]
    home = Path.home()
    chosen = [name for name in _AGENTS if getattr(args, name)]
    if chosen:
        return [home / _AGENTS[name] / "skills" for name in chosen]
    found = [
        home / folder / "skills"
        for folder in _AGENTS.values()
        if (home / folder).is_dir()
    ]
    if not found:
        raise ValueError(
            f"No agent found: neither {home / '.claude'} nor {home / '.codex'} "
            "exists. Use --claude or --codex to create one, or --dir DIR."
        )
    return found


def _is_dftk_skill(folder: Path) -> bool:
    skill = folder / "SKILL.md"
    if not skill.is_file():
        return False
    head = skill.read_text(errors="replace").split("\n---", 1)[0]
    return any(line.strip() == "name: dftk" for line in head.splitlines())


def _install(skills_folder: Path) -> bool:
    """Copy the skill to *skills_folder*/dftk; return True if it replaced one."""
    dest = skills_folder / "dftk"
    if dest.is_symlink():
        raise ValueError(
            f"{dest} is a symbolic link; remove it first if dftk should "
            "install a copy there."
        )
    replaced = dest.exists()
    if replaced:
        if not _is_dftk_skill(dest):
            raise ValueError(
                f"{dest} exists but is not the dftk skill; not replacing it."
            )
        shutil.rmtree(dest)
    skills_folder.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SKILL_DIR, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    return replaced


def _skill(args: argparse.Namespace) -> None:
    if args.action == "path":
        print(SKILL_DIR)
        return
    folders = _skills_folders(args)
    for folder in folders:
        replaced = _install(folder)
        verb = "Updated" if replaced else "Installed"
        print(f"{verb} the dftk skill (dftk {_dftk_version()}) in {folder / 'dftk'}")
    print('Run "dftk skill install" again after upgrading dftk.')


class SkillCommand(BaseCommand):
    @property
    def name(self) -> str:
        return "skill"

    @property
    def help(self) -> str:
        return "Install the agent skill for Claude Code, Codex and others"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.formatter_class = argparse.RawDescriptionHelpFormatter
        parser.epilog = _EPILOG

        parser.add_argument(
            "action",
            choices=["install", "path"],
            help="install: copy the skill to the agents' skills folders; "
            "path: print where the packaged skill is.",
        )
        g = parser.add_argument_group("install options")
        g.add_argument(
            "--claude",
            action="store_true",
            help="Install into ~/.claude/skills (created if missing).",
        )
        g.add_argument(
            "--codex",
            action="store_true",
            help="Install into ~/.codex/skills (created if missing).",
        )
        g.add_argument(
            "--dir",
            default=None,
            metavar="DIR",
            help="Install into DIR/dftk instead.",
        )

    def execute(self, args: argparse.Namespace) -> None:
        _skill(args)
