"""
CLI integration tests for dftk.

These tests invoke the package via `python -m dftk.cli` so they exercise
argument parsing, command dispatch, and I/O in a real subprocess.  Each test
feeds TSV on stdin and parses TSV from stdout, making them independent of any
installed entry-point.
"""

import io
import re
import subprocess
import sys

import pandas as pd
import pytest


def _run(args: list[str], stdin_tsv: str = "") -> pd.DataFrame:
    """Run `dftk <args>` with stdin_tsv fed to stdin.

    Returns the parsed TSV output as a DataFrame.
    Raises AssertionError if the process exits non-zero.
    """
    result = subprocess.run(
        [sys.executable, "-m", "dftk.cli"] + args,
        input=stdin_tsv.encode(),
        capture_output=True,
    )
    assert result.returncode == 0, (
        f"dftk exited {result.returncode}\n"
        f"stderr: {result.stderr.decode()}"
    )
    return pd.read_csv(io.BytesIO(result.stdout), sep="\t")


# ---------------------------------------------------------------------------
# Simple TSV payload reused across tests
# ---------------------------------------------------------------------------

_SIMPLE_TSV = "x\ty\n1\t10\n2\t20\n3\t30\n4\t40\n5\t50\n"
_GROUPED_TSV = "group\tvalue\nA\t1\nA\t2\nA\t3\nB\t10\nB\t20\nB\t30\n"


# ---------------------------------------------------------------------------
# stat subcommand
# ---------------------------------------------------------------------------


class TestStatCLI:

    def test_single_col_schema(self):
        df = _run(["stat", "-", "-c", "x"], _SIMPLE_TSV)
        assert "name" in df.columns
        assert "mean" in df.columns
        assert "count" in df.columns

    def test_single_col_values(self):
        df = _run(["stat", "-", "-c", "x"], _SIMPLE_TSV)
        assert df["mean"].iloc[0] == pytest.approx(3.0)
        assert df["count"].iloc[0] == 5

    def test_multi_col_two_rows(self):
        df = _run(["stat", "-", "-c", "x", "y"], _SIMPLE_TSV)
        assert len(df) == 2
        assert set(df["name"]) == {"x", "y"}

    def test_grouped_schema(self):
        df = _run(["stat", "-", "-c", "value", "-g", "group"], _GROUPED_TSV)
        assert "group" in df.columns
        assert "name" in df.columns

    def test_grouped_values(self):
        df = _run(["stat", "-", "-c", "value", "-g", "group"], _GROUPED_TSV)
        a_mean = df[df["group"] == "A"]["mean"].iloc[0]
        b_mean = df[df["group"] == "B"]["mean"].iloc[0]
        assert a_mean == pytest.approx(2.0)
        assert b_mean == pytest.approx(20.0)

    def test_bootstrap_samplenum(self):
        """Bootstrap mode emits a samplenum column with the right values."""
        df = _run(
            ["stat", "-", "-c", "x", "--bootstrap", "3", "--randomseed", "42"],
            _SIMPLE_TSV,
        )
        assert "samplenum" in df.columns
        assert set(df["samplenum"]) == {1, 2, 3}

    def test_bootstrap_with_groupcol_keeps_group_column(self):
        """Regression: pandas 3.0's groupby(...).apply() used to silently drop
        the grouping column, crashing -g combined with --bootstrap."""
        df = _run(
            [
                "stat",
                "-",
                "-c",
                "value",
                "-g",
                "group",
                "--bootstrap",
                "3",
                "--randomseed",
                "42",
            ],
            _GROUPED_TSV,
        )
        assert "group" in df.columns
        assert set(df["group"]) == {"A", "B"}
        assert len(df) == 6  # 2 groups x 3 bootstrap samples

    def test_confidence_level(self):
        """--confidencelevel changes cilo/cihi bounds."""
        df_95 = _run(["stat", "-", "-c", "x"], _SIMPLE_TSV)
        df_50 = _run(["stat", "-", "-c", "x", "--confidencelevel", "50"], _SIMPLE_TSV)
        # 50% CI is strictly narrower than 95% CI
        assert df_50["cilo"].iloc[0] > df_95["cilo"].iloc[0]
        assert df_50["cihi"].iloc[0] < df_95["cihi"].iloc[0]


# ---------------------------------------------------------------------------
# print subcommand
# ---------------------------------------------------------------------------


class TestPrintCLI:

    def test_passthrough(self):
        df = _run(["print", "-"], _SIMPLE_TSV)
        assert list(df.columns) == ["x", "y"]
        assert len(df) == 5


# ---------------------------------------------------------------------------
# help subcommand
# ---------------------------------------------------------------------------


class TestHelpCLI:

    def test_help_lists_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "dftk.cli", "help"],
            capture_output=True,
        )
        assert result.returncode == 0
        output = result.stdout.decode()
        assert "stat" in output
        assert "print" in output

    def test_help_stat(self):
        result = subprocess.run(
            [sys.executable, "-m", "dftk.cli", "help", "stat"],
            capture_output=True,
        )
        assert result.returncode == 0
        output = result.stdout.decode()
        assert "--cols" in output or "-c" in output


# ---------------------------------------------------------------------------
# broken pipe (e.g. `dftk ... | head -1`)
# ---------------------------------------------------------------------------


class TestBrokenPipeCLI:

    @pytest.mark.parametrize("command", ["print", "query"])
    def test_reader_closing_early_exits_141_quietly(self, command, tmp_path):
        # Well past the 64 KB pipe buffer, so the write is still in progress
        # when the reader goes away.
        path = tmp_path / "big.tsv"
        path.write_text("x\n" + "".join(f"{i}\n" for i in range(100_000)))

        proc = subprocess.Popen(
            [sys.executable, "-m", "dftk.cli", command, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout.readline() == b"x\n"
        proc.stdout.close()
        stderr = proc.stderr.read().decode()
        proc.wait()

        assert proc.returncode == 141, stderr
        assert stderr == ""


# ---------------------------------------------------------------------------
# lazy command imports: a command must not pay for other commands' dependencies
# ---------------------------------------------------------------------------

_HEAVY_MODULES = ["scipy.stats", "scipy.special", "statsmodels", "matplotlib"]

# Commands that need none of the heavy modules just to start up.
_LIGHT_COMMANDS = (
    "eval query merge concat melt pivot func interp binx transpose segid "
    "stat describe info dataset sample split annotate print clean"
).split()


def _heavy_modules_loaded(argv: list[str]) -> list[str]:
    """Run `dftk <argv>` in a fresh interpreter; return heavy modules it imported."""
    code = (
        "import sys\n"
        f"sys.argv = ['dftk', *{argv!r}]\n"
        "from dftk.cli import main\n"
        "try:\n"
        "    main()\n"
        "except SystemExit:\n"
        "    pass\n"
        f"loaded = [m for m in {_HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(','.join(loaded), file=sys.stderr)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    last_line = result.stderr.decode().strip().splitlines()[-1:]
    return [m for m in "".join(last_line).split(",") if m]


class TestLazyImportsCLI:

    @pytest.mark.parametrize("command", _LIGHT_COMMANDS)
    def test_light_command_startup_skips_heavy_modules(self, command):
        assert _heavy_modules_loaded([command, "--help"]) == []

    def test_eval_plain_formula_skips_scipy(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text(_SIMPLE_TSV)
        assert _heavy_modules_loaded(["eval", str(path), "-f", "z = x + y"]) == []

    def test_eval_special_function_still_works(self, tmp_path):
        path = tmp_path / "data.tsv"
        path.write_text(_SIMPLE_TSV)
        result = subprocess.run(
            [sys.executable, "-m", "dftk.cli", "eval", str(path), "-f", "z = gamma(x)"],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr.decode()
        df = pd.read_csv(io.BytesIO(result.stdout), sep="\t")
        assert df["z"].tolist() == pytest.approx([1.0, 1.0, 2.0, 6.0, 24.0])

    def test_unknown_command_lists_all_commands(self):
        result = subprocess.run(
            [sys.executable, "-m", "dftk.cli", "nosuch"], capture_output=True
        )
        assert result.returncode == 2
        stderr = result.stderr.decode()
        assert "invalid choice: 'nosuch'" in stderr
        # Python 3.12 quotes the choices ('fit', …), 3.14 doesn't (fit, …).
        choices = re.findall(r"'?(\w+)'?(?:,|\))", stderr.split("choose from")[1])
        assert {"eval", "fit", "scat", "help"} <= set(choices)
