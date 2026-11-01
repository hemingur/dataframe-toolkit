"""Tests for dftk.commands.skill_cmd (dftk skill install / path)."""

import filecmp
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from dftk.commands.skill_cmd import SKILL_DIR, _skill
from tests.conftest import make_args


def _args(**kwargs):
    defaults = dict(action="install", claude=False, codex=False, dir=None)
    defaults.update(kwargs)
    return make_args(**defaults)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _same_as_packaged(folder: Path) -> bool:
    files = sorted(p.relative_to(SKILL_DIR) for p in SKILL_DIR.rglob("*.md"))
    copied = sorted(p.relative_to(folder) for p in folder.rglob("*") if p.is_file())
    if files != copied:
        return False
    return all(filecmp.cmp(SKILL_DIR / f, folder / f, shallow=False) for f in files)


class TestPackagedSkill:
    def test_skill_files_exist(self):
        assert (SKILL_DIR / "SKILL.md").is_file()
        assert (SKILL_DIR / "references" / "snakemake.md").is_file()

    def test_path_prints_packaged_folder(self, capsys):
        _skill(_args(action="path"))
        assert capsys.readouterr().out.strip() == str(SKILL_DIR)


class TestInstallTargets:
    def test_installs_where_claude_exists(self, home, capsys):
        (home / ".claude").mkdir()
        _skill(_args())
        assert _same_as_packaged(home / ".claude" / "skills" / "dftk")
        assert not (home / ".codex").exists()
        out = capsys.readouterr().out
        assert "Installed the dftk skill (dftk " in out
        assert str(home / ".claude" / "skills" / "dftk") in out

    def test_installs_for_both_agents(self, home, capsys):
        (home / ".claude").mkdir()
        (home / ".codex").mkdir()
        _skill(_args())
        assert _same_as_packaged(home / ".claude" / "skills" / "dftk")
        assert _same_as_packaged(home / ".codex" / "skills" / "dftk")
        assert capsys.readouterr().out.count("Installed") == 2

    def test_no_agent_found_is_an_error(self, home):
        with pytest.raises(ValueError, match="--claude or --codex"):
            _skill(_args())

    def test_codex_option_creates_its_folder(self, home):
        _skill(_args(codex=True))
        assert _same_as_packaged(home / ".codex" / "skills" / "dftk")
        assert not (home / ".claude").exists()

    def test_option_installs_only_that_agent(self, home):
        (home / ".claude").mkdir()
        (home / ".codex").mkdir()
        _skill(_args(claude=True))
        assert (home / ".claude" / "skills" / "dftk").is_dir()
        assert not (home / ".codex" / "skills").exists()

    def test_dir_option(self, home, tmp_path):
        target = tmp_path / "agents" / "skills"
        _skill(_args(dir=str(target)))
        assert _same_as_packaged(target / "dftk")


class TestReinstall:
    def test_second_install_says_updated(self, home, capsys):
        (home / ".claude").mkdir()
        _skill(_args())
        _skill(_args())
        out = capsys.readouterr().out
        assert "Installed" in out.splitlines()[0]
        assert "Updated" in out

    def test_old_files_are_removed(self, home):
        (home / ".claude").mkdir()
        _skill(_args())
        stale = home / ".claude" / "skills" / "dftk" / "references" / "old.md"
        stale.write_text("from an older dftk")
        _skill(_args())
        assert not stale.exists()
        assert _same_as_packaged(home / ".claude" / "skills" / "dftk")

    def test_other_skills_are_left_alone(self, home):
        other = home / ".claude" / "skills" / "mine"
        other.mkdir(parents=True)
        (other / "SKILL.md").write_text("---\nname: mine\n---\n")
        _skill(_args())
        assert (other / "SKILL.md").read_text() == "---\nname: mine\n---\n"

    def test_refuses_to_replace_a_folder_that_is_not_the_dftk_skill(self, home):
        dest = home / ".claude" / "skills" / "dftk"
        dest.mkdir(parents=True)
        (dest / "notes.txt").write_text("keep me")
        with pytest.raises(ValueError, match="not the dftk skill"):
            _skill(_args())
        assert (dest / "notes.txt").read_text() == "keep me"

    def test_refuses_to_replace_a_symlink(self, home, tmp_path):
        repo_copy = tmp_path / "repo_skill"
        shutil.copytree(SKILL_DIR, repo_copy)
        skills = home / ".claude" / "skills"
        skills.mkdir(parents=True)
        (skills / "dftk").symlink_to(repo_copy)
        with pytest.raises(ValueError, match="symbolic link"):
            _skill(_args())
        assert (repo_copy / "SKILL.md").is_file()


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build")
def test_wheel_contains_the_skill(tmp_path):
    root = Path(__file__).parents[2]
    subprocess.run(
        ["uv", "build", "--wheel", "--quiet", "--out-dir", str(tmp_path)],
        cwd=root,
        check=True,
    )
    (wheel,) = tmp_path.glob("*.whl")
    names = set(zipfile.ZipFile(wheel).namelist())
    for f in SKILL_DIR.rglob("*.md"):
        assert f"dftk/skills/dftk/{f.relative_to(SKILL_DIR).as_posix()}" in names
