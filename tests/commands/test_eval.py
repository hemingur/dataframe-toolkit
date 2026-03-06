"""
Tests for stattools.commands.eval_cmd._eval and helper functions.
"""

import numpy as np
import pandas as pd
import pytest

from stattools.commands.eval_cmd import _eval, _parse_formula, _special_function, _string_function
from tests.conftest import make_args


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def num_df():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def str_df():
    return pd.DataFrame({
        "name": ["Alice", "BOB", "charlie"],
        "tag":  ["foo:bar", "baz:qux", "no-sep"],
    })


@pytest.fixture
def seq_df():
    return pd.DataFrame({
        "seq": ["ATGCATGC", "GGGGCCCC", "AAAAATTT"],
    })


def _eval_args(**overrides):
    defaults = dict(
        constant=[],
        formula=[],
        strfunc=[],
        joinsep="",
        intbool=False,
    )
    defaults.update(overrides)
    return make_args(**defaults)


# ---------------------------------------------------------------------------
# _parse_formula
# ---------------------------------------------------------------------------


class TestParseFormula:

    def test_basic(self):
        dest, func, cols, body = _parse_formula("z = add(x, y)")
        assert dest == "z"
        assert func == "add"
        assert cols == ["x", "y"]
        assert body == "x, y"

    def test_single_col(self):
        dest, func, cols, body = _parse_formula("out = lower(name)")
        assert dest == "out"
        assert func == "lower"
        assert cols == ["name"]

    def test_bad_formula_raises(self):
        with pytest.raises(ValueError, match="Bad formula"):
            _parse_formula("no equals sign")


# ---------------------------------------------------------------------------
# Constant columns
# ---------------------------------------------------------------------------


class TestConstantColumns:

    def test_integer_constant(self, num_df):
        result = _eval(num_df.copy(), _eval_args(constant=["batch = 3"]))
        assert "batch" in result.columns
        assert (result["batch"] == 3).all()
        assert result["batch"].dtype in (np.int64, np.int32, int)

    def test_float_constant(self, num_df):
        result = _eval(num_df.copy(), _eval_args(constant=["scale = 1.5"]))
        assert result["scale"].iloc[0] == pytest.approx(1.5)

    def test_string_constant(self, num_df):
        result = _eval(num_df.copy(), _eval_args(constant=["label = control"]))
        assert result["label"].iloc[0] == "control"

    def test_multiple_constants(self, num_df):
        result = _eval(num_df.copy(), _eval_args(constant=["a = 1", "b = 2"]))
        assert "a" in result.columns
        assert "b" in result.columns


# ---------------------------------------------------------------------------
# Formula evaluation (-f)
# ---------------------------------------------------------------------------


class TestFormulaEval:

    def test_arithmetic(self, num_df):
        result = _eval(num_df.copy(), _eval_args(formula=["z = x + y"]))
        assert list(result["z"]) == pytest.approx([11.0, 22.0, 33.0])

    def test_boolean_formula(self, num_df):
        result = _eval(num_df.copy(), _eval_args(formula=["flag = x > 1"]))
        assert list(result["flag"]) == [False, True, True]

    def test_intbool_conversion(self, num_df):
        result = _eval(
            num_df.copy(),
            _eval_args(formula=["flag = x > 1"], intbool=True),
        )
        assert list(result["flag"]) == [0, 1, 1]
        assert result["flag"].dtype == np.int64

    def test_multiple_formulas_chain(self, num_df):
        result = _eval(
            num_df.copy(),
            _eval_args(formula=["z = x + y", "w = z * 2"]),
        )
        assert list(result["w"]) == pytest.approx([22.0, 44.0, 66.0])

    def test_special_function_sign(self, num_df):
        df = pd.DataFrame({"x": [-3.0, 0.0, 5.0]})
        result = _eval(df, _eval_args(formula=["s = sign(x)"]))
        assert list(result["s"]) == pytest.approx([-1.0, 0.0, 1.0])

    def test_special_function_overlap(self):
        df = pd.DataFrame({
            "lo1": [1], "hi1": [5], "lo2": [3], "hi2": [7],
        })
        result = _eval(df, _eval_args(formula=["ov = overlap(lo1, hi1, lo2, hi2)"]))
        assert result["ov"].iloc[0] == 3  # overlap of [1,5] and [3,7] = 3

    def test_special_function_colsum(self):
        df = pd.DataFrame({"a1": [1.0], "a2": [2.0], "a3": [3.0], "b": [99.0]})
        result = _eval(df, _eval_args(formula=["total = colsum(a*)"] ))
        assert result["total"].iloc[0] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# String functions (-s)
# ---------------------------------------------------------------------------


class TestStringFunctions:

    def test_lower(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["lname = lower(name)"]))
        assert list(result["lname"]) == ["alice", "bob", "charlie"]

    def test_upper(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["uname = upper(name)"]))
        assert result["uname"].iloc[0] == "ALICE"

    def test_len(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["n = len(name)"]))
        assert result["n"].iloc[0] == 5   # "Alice"
        assert result["n"].iloc[1] == 3   # "BOB"

    def test_replace(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["t2 = replace(tag, :, _)"]))
        assert result["t2"].iloc[0] == "foo_bar"

    def test_leftsplit_left_part(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["left = leftsplit(tag, :, 0)"]))
        assert result["left"].iloc[0] == "foo"
        assert result["left"].iloc[2] == "no-sep"  # no separator → whole string

    def test_leftsplit_right_part(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["right = leftsplit(tag, :, 1)"]))
        assert result["right"].iloc[0] == "bar"
        assert result["right"].iloc[2] == ""  # no separator → empty

    def test_rightsplit(self, str_df):
        df = pd.DataFrame({"s": ["a.b.c"]})
        result = _eval(df, _eval_args(strfunc=["stem = rightsplit(s, ., 0)"]))
        assert result["stem"].iloc[0] == "a.b"

    def test_substr(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["s = substr(name, 0, 3)"]))
        assert result["s"].iloc[0] == "Ali"

    def test_join(self, num_df):
        result = _eval(
            num_df.copy(),
            _eval_args(strfunc=["combo = join(x, y)"], joinsep="-"),
        )
        assert result["combo"].iloc[0] == "1.0-10.0"

    def test_find(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["pos = find(tag, :)"]))
        assert result["pos"].iloc[0] == 3  # "foo:bar"

    def test_md5_length(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["h = md5(name)"]))
        assert len(result["h"].iloc[0]) == 32

    def test_md5_deterministic(self, str_df):
        r1 = _eval(str_df.copy(), _eval_args(strfunc=["h = md5(name)"]))
        r2 = _eval(str_df.copy(), _eval_args(strfunc=["h = md5(name)"]))
        assert list(r1["h"]) == list(r2["h"])


# ---------------------------------------------------------------------------
# DNA / sequence functions
# ---------------------------------------------------------------------------


class TestSeqFunctions:

    def test_gccontent(self, seq_df):
        result = _eval(seq_df.copy(), _eval_args(strfunc=["gc = gccontent(seq)"]))
        assert result["gc"].iloc[0] == pytest.approx(50.0)   # ATGCATGC
        assert result["gc"].iloc[1] == pytest.approx(100.0)  # GGGGCCCC

    def test_basecount(self, seq_df):
        result = _eval(seq_df.copy(), _eval_args(strfunc=["n = basecount(seq)"]))
        assert result["n"].iloc[0] == 8

    def test_motifcount(self, seq_df):
        result = _eval(seq_df.copy(), _eval_args(strfunc=["m = motifcount(ATG, seq)"]))
        assert result["m"].iloc[0] == 2   # ATGCATGC has two ATG

    def test_isfile_existing(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        df = pd.DataFrame({"path": [str(f), "/nonexistent/file"]})
        result = _eval(df, _eval_args(strfunc=["ok = isfile(path)"]))
        assert result["ok"].iloc[0] == 1
        assert result["ok"].iloc[1] == 0

    def test_stem(self, tmp_path):
        df = pd.DataFrame({"path": ["/some/dir/myfile.tsv"]})
        result = _eval(df, _eval_args(strfunc=["s = stem(path)"]))
        assert result["s"].iloc[0] == "myfile"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_dataframe_does_not_crash(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        result = _eval(df, _eval_args(formula=["y = x + 1"]))
        assert "y" in result.columns
        assert len(result) == 0

    def test_formula_and_constant_together(self, num_df):
        result = _eval(
            num_df.copy(),
            _eval_args(constant=["scale = 2"], formula=["z = x * scale"]),
        )
        assert list(result["z"]) == pytest.approx([2.0, 4.0, 6.0])
