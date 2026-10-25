"""
Tests for dftk.commands.eval_cmd._eval and helper functions.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import special as _sp

from dftk.commands.eval_cmd import _eval
from dftk.common.evalfuncs import parse_formula as _parse_formula
from tests.conftest import make_args

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def num_df():
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0],
            "y": [10.0, 20.0, 30.0],
        }
    )


@pytest.fixture
def str_df():
    return pd.DataFrame(
        {
            "name": ["Alice", "BOB", "charlie"],
            "tag": ["foo:bar", "baz:qux", "no-sep"],
        }
    )


@pytest.fixture
def seq_df():
    return pd.DataFrame(
        {
            "seq": ["ATGCATGC", "GGGGCCCC", "AAAAATTT"],
        }
    )


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
        df = pd.DataFrame(
            {
                "lo1": [1],
                "hi1": [5],
                "lo2": [3],
                "hi2": [7],
            }
        )
        result = _eval(df, _eval_args(formula=["ov = overlap(lo1, hi1, lo2, hi2)"]))
        assert result["ov"].iloc[0] == 3  # overlap of [1,5] and [3,7] = 3

    def test_special_function_colsum(self):
        df = pd.DataFrame({"a1": [1.0], "a2": [2.0], "a3": [3.0], "b": [99.0]})
        result = _eval(df, _eval_args(formula=["total = colsum(a*)"]))
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
        assert result["n"].iloc[0] == 5  # "Alice"
        assert result["n"].iloc[1] == 3  # "BOB"

    def test_replace(self, str_df):
        result = _eval(str_df.copy(), _eval_args(strfunc=["t2 = replace(tag, :, _)"]))
        assert result["t2"].iloc[0] == "foo_bar"

    def test_leftsplit_left_part(self, str_df):
        result = _eval(
            str_df.copy(), _eval_args(strfunc=["left = leftsplit(tag, :, 0)"])
        )
        assert result["left"].iloc[0] == "foo"
        assert result["left"].iloc[2] == "no-sep"  # no separator → whole string

    def test_leftsplit_right_part(self, str_df):
        result = _eval(
            str_df.copy(), _eval_args(strfunc=["right = leftsplit(tag, :, 1)"])
        )
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
        assert result["gc"].iloc[0] == pytest.approx(50.0)  # ATGCATGC
        assert result["gc"].iloc[1] == pytest.approx(100.0)  # GGGGCCCC

    def test_basecount(self, seq_df):
        result = _eval(seq_df.copy(), _eval_args(strfunc=["n = basecount(seq)"]))
        assert result["n"].iloc[0] == 8

    def test_motifcount(self, seq_df):
        result = _eval(seq_df.copy(), _eval_args(strfunc=["m = motifcount(ATG, seq)"]))
        assert result["m"].iloc[0] == 2  # ATGCATGC has two ATG

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


# ---------------------------------------------------------------------------
# scipy.special and extended numpy functions
# ---------------------------------------------------------------------------


@pytest.fixture
def sp_df():
    return pd.DataFrame(
        {"x": [0.5, 1.0, 2.0], "a": [1.0, 2.0, 3.0], "b": [2.0, 3.0, 4.0]}
    )


class TestScipySpecial:
    # -- log-gamma / digamma ---------------------------------------------------

    def test_gamma(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["g = gamma(x)"]))
        expected = _sp.gamma(np.array([0.5, 1.0, 2.0]))
        assert list(result["g"]) == pytest.approx(list(expected))

    def test_lgamma(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["lg = lgamma(x)"]))
        expected = _sp.gammaln(np.array([0.5, 1.0, 2.0]))
        assert list(result["lg"]) == pytest.approx(list(expected))

    def test_digamma(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["dg = digamma(x)"]))
        expected = _sp.digamma(np.array([0.5, 1.0, 2.0]))
        assert list(result["dg"]) == pytest.approx(list(expected))

    def test_beta_two_columns(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["bt = beta(a, b)"]))
        expected = [_sp.beta(a, b) for a, b in zip([1, 2, 3], [2, 3, 4], strict=False)]
        assert list(result["bt"]) == pytest.approx(expected)

    def test_betaln_two_columns(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["bl = betaln(a, b)"]))
        expected = [
            _sp.betaln(a, b) for a, b in zip([1, 2, 3], [2, 3, 4], strict=False)
        ]
        assert list(result["bl"]) == pytest.approx(expected)

    # -- error functions -------------------------------------------------------

    def test_erf(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["e = erf(x)"]))
        expected = _sp.erf(np.array([0.5, 1.0, 2.0]))
        assert list(result["e"]) == pytest.approx(list(expected))

    def test_erfc(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["ec = erfc(x)"]))
        expected = _sp.erfc(np.array([0.5, 1.0, 2.0]))
        assert list(result["ec"]) == pytest.approx(list(expected))

    def test_erfinv(self, sp_df):
        # erfinv(erf(x)) == x
        df = pd.DataFrame({"x": [0.1, 0.5, 0.9]})
        r1 = _eval(df.copy(), _eval_args(formula=["e = erf(x)"]))
        r2 = _eval(r1.copy(), _eval_args(formula=["xi = erfinv(e)"]))
        assert list(r2["xi"]) == pytest.approx([0.1, 0.5, 0.9])

    # -- sigmoid / logit -------------------------------------------------------

    def test_expit(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["s = expit(x)"]))
        expected = _sp.expit(np.array([0.5, 1.0, 2.0]))
        assert list(result["s"]) == pytest.approx(list(expected))

    def test_logit_expit_roundtrip(self, sp_df):
        # logit(expit(x)) == x
        r1 = _eval(sp_df.copy(), _eval_args(formula=["s = expit(x)"]))
        r2 = _eval(r1.copy(), _eval_args(formula=["xi = logit(s)"]))
        assert list(r2["xi"]) == pytest.approx([0.5, 1.0, 2.0])

    # -- Bessel fixed order ----------------------------------------------------

    def test_j0(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["j = j0(x)"]))
        expected = _sp.j0(np.array([0.5, 1.0, 2.0]))
        assert list(result["j"]) == pytest.approx(list(expected))

    def test_j1(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["j = j1(x)"]))
        expected = _sp.j1(np.array([0.5, 1.0, 2.0]))
        assert list(result["j"]) == pytest.approx(list(expected))

    def test_y0(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["y = y0(x)"]))
        expected = _sp.y0(np.array([0.5, 1.0, 2.0]))
        assert list(result["y"]) == pytest.approx(list(expected))

    def test_i0(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["i = i0(x)"]))
        expected = _sp.i0(np.array([0.5, 1.0, 2.0]))
        assert list(result["i"]) == pytest.approx(list(expected))

    def test_k0(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["k = k0(x)"]))
        expected = _sp.k0(np.array([0.5, 1.0, 2.0]))
        assert list(result["k"]) == pytest.approx(list(expected))

    # -- Bessel parametric order -----------------------------------------------

    def test_jn_order_2(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["j = jn(2, x)"]))
        expected = _sp.jn(2, np.array([0.5, 1.0, 2.0]))
        assert list(result["j"]) == pytest.approx(list(expected))

    def test_jn_order_0_matches_j0(self, sp_df):
        r_jn = _eval(sp_df.copy(), _eval_args(formula=["j = jn(0, x)"]))
        r_j0 = _eval(sp_df.copy(), _eval_args(formula=["j = j0(x)"]))
        assert list(r_jn["j"]) == pytest.approx(list(r_j0["j"]))

    def test_yn_order_1_matches_y1(self, sp_df):
        r_yn = _eval(sp_df.copy(), _eval_args(formula=["y = yn(1, x)"]))
        r_y1 = _eval(sp_df.copy(), _eval_args(formula=["y = y1(x)"]))
        assert list(r_yn["y"]) == pytest.approx(list(r_y1["y"]))

    def test_iv_fractional_order(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["iv = iv(1.5, x)"]))
        expected = _sp.iv(1.5, np.array([0.5, 1.0, 2.0]))
        assert list(result["iv"]) == pytest.approx(list(expected))

    def test_kv_order_2(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["k = kv(2, x)"]))
        expected = _sp.kv(2, np.array([0.5, 1.0, 2.0]))
        assert list(result["k"]) == pytest.approx(list(expected))

    def test_jn_bad_order_raises(self, sp_df):
        with pytest.raises((ValueError, KeyError)):
            _eval(sp_df.copy(), _eval_args(formula=["j = jn(xcol, x)"]))

    # -- extended numpy functions ----------------------------------------------

    def test_cbrt(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["c = cbrt(x)"]))
        expected = np.cbrt([0.5, 1.0, 2.0])
        assert list(result["c"]) == pytest.approx(list(expected))

    def test_log1p(self, sp_df):
        result = _eval(sp_df.copy(), _eval_args(formula=["l = log1p(x)"]))
        expected = np.log1p([0.5, 1.0, 2.0])
        assert list(result["l"]) == pytest.approx(list(expected))

    def test_expm1_log1p_roundtrip(self, sp_df):
        r1 = _eval(sp_df.copy(), _eval_args(formula=["lp = log1p(x)"]))
        r2 = _eval(r1.copy(), _eval_args(formula=["xi = expm1(lp)"]))
        assert list(r2["xi"]) == pytest.approx([0.5, 1.0, 2.0])


# ---------------------------------------------------------------------------
# Filling missing values with -f (pandas methods)
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_df():
    # Two sites, sorted by day, with gaps at the start, middle and end
    return pd.DataFrame(
        {
            "site": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "day": [1, 2, 3, 4, 1, 2, 3, 4],
            "temp": [10.0, None, 14.0, None, None, 20.0, None, 26.0],
        }
    )


def _values(s: pd.Series) -> list:
    return [None if pd.isna(v) else v for v in s]


class TestFillFormulas:
    def test_fillna_float_from_nullable_int(self):
        # What a left merge gives: float64 column, Int64 defaults (_r)
        df = pd.DataFrame(
            {
                "depth": [5.0, None, None, None],
                "depth_r": pd.array([10, 10, 20, None], dtype="Int64"),
            }
        )
        result = _eval(df, _eval_args(formula=["depth = depth.fillna(depth_r)"]))
        assert _values(result["depth"]) == [5.0, 10.0, 20.0, None]

    def test_fillna_into_new_column_keeps_original(self):
        df = pd.DataFrame({"x": [1.0, None], "y": [7.0, 8.0]})
        result = _eval(df, _eval_args(formula=["z = x.fillna(y)"]))
        assert _values(result["x"]) == [1.0, None]
        assert _values(result["z"]) == [1.0, 8.0]

    def test_chained_fillna_with_literal(self):
        df = pd.DataFrame({"x": [None, None, 3.0], "y": [5.0, None, 6.0]})
        result = _eval(df, _eval_args(formula=["x = x.fillna(y).fillna(0)"]))
        assert _values(result["x"]) == [5.0, 0.0, 3.0]

    def test_fillna_text_from_other_column(self):
        df = pd.DataFrame({"gene_id": ["E1", "E2"], "symbol": ["BRCA1", None]})
        result = _eval(df, _eval_args(formula=["symbol = symbol.fillna(gene_id)"]))
        assert list(result["symbol"]) == ["BRCA1", "E2"]

    def test_overwrite_with_boolean_result(self):
        df = pd.DataFrame({"x": [1.0, None]})
        result = _eval(df, _eval_args(formula=["x = x.isna()"]))
        assert list(result["x"]) == [False, True]

    def test_ffill_crosses_groups_without_groupby(self, ts_df):
        result = _eval(ts_df, _eval_args(formula=["temp = temp.ffill()"]))
        assert result["temp"].iloc[4] == 14.0  # B day 1 took A's last value

    def test_grouped_ffill_and_bfill(self, ts_df):
        result = _eval(
            ts_df,
            _eval_args(
                formula=[
                    "f = temp.groupby(site).ffill()",
                    "b = temp.groupby(site).bfill()",
                ]
            ),
        )
        assert _values(result["f"]) == [10, 10, 14, 14, None, 20, 20, 26]
        assert _values(result["b"]) == [10, 14, 14, None, 20, 20, 26, 26]

    def test_grouped_ffill_then_bfill_fills_both_ends(self, ts_df):
        formula = "temp = temp.groupby(site).ffill().fillna(temp.groupby(site).bfill())"
        result = _eval(ts_df, _eval_args(formula=[formula]))
        assert _values(result["temp"]) == [10, 10, 14, 14, 20, 20, 20, 26]

    def test_grouped_ffill_follows_row_order(self, ts_df):
        # Unsorted rows: A day 4 is filled from the row above it (day 1)
        df = ts_df.iloc[[0, 3, 2, 1]].reset_index(drop=True)
        result = _eval(df, _eval_args(formula=["temp = temp.groupby(site).ffill()"]))
        assert _values(result["day"]) == [1, 4, 3, 2]
        assert _values(result["temp"]) == [10, 10, 14, 14]

    def test_fillna_with_group_mean(self, ts_df):
        formula = "temp = temp.fillna(temp.groupby(site).transform('mean'))"
        result = _eval(ts_df, _eval_args(formula=[formula]))
        assert _values(result["temp"]) == [10, 12, 14, 12, 23, 20, 23, 26]

    def test_interpolate_is_by_row_position(self):
        df = pd.DataFrame({"x": [0.0, None, None, 6.0]})
        result = _eval(df, _eval_args(formula=["x = x.interpolate()"]))
        assert _values(result["x"]) == [0.0, 2.0, 4.0, 6.0]

    def test_parquet_annotations_are_kept(self):
        df = pd.DataFrame({"x": [1.0, None], "y": [2.0, 3.0]})
        df.attrs["_parquet_meta"] = {"source": "test"}
        result = _eval(df, _eval_args(formula=["x = x.fillna(y)"]))
        assert result.attrs["_parquet_meta"] == {"source": "test"}


class TestFormulaErrors:
    def test_text_method_without_str_suggests_str(self, str_df):
        with pytest.raises(ValueError, match=r"name\.str\.lower\(\)"):
            _eval(str_df, _eval_args(formula=["n = name.lower()"]))

    def test_unknown_method_points_to_str(self, str_df):
        with pytest.raises(ValueError, match="no method 'tolower'.*name.str.lower"):
            _eval(str_df, _eval_args(formula=["n = name.tolower()"]))

    def test_unknown_column_is_input_error(self, num_df):
        with pytest.raises(ValueError, match="'nope' is not defined"):
            _eval(num_df, _eval_args(formula=["z = nope + 1"]))

    def test_wrong_argument_is_input_error(self, num_df):
        with pytest.raises(ValueError, match="z = x.fillna.*unexpected keyword"):
            _eval(num_df, _eval_args(formula=["z = x.fillna(foo=1)"]))

    def test_str_on_numbers_is_input_error(self, num_df):
        with pytest.raises(ValueError, match="only use .str accessor"):
            _eval(num_df, _eval_args(formula=["z = x.str.lower()"]))

    def test_method_call_reports_pandas_error(self, ts_df):
        formula = "temp = temp.groupby(site).transform('interpolate')"
        with pytest.raises(ValueError, match="not a valid function name") as exc:
            _eval(ts_df, _eval_args(formula=[formula]))
        assert "special function" not in str(exc.value)

    def test_unknown_function(self, num_df):
        with pytest.raises(ValueError, match="unknown function 'frobnicate'"):
            _eval(num_df, _eval_args(formula=["z = frobnicate(x)"]))

    def test_special_functions_still_work(self, num_df):
        result = _eval(num_df, _eval_args(formula=["z = sum(x, y)"]))
        assert list(result["z"]) == pytest.approx([11.0, 22.0, 33.0])

    def test_empty_table_still_reports_unknown_method(self):
        df = pd.DataFrame({"x": pd.Series([], dtype=float)})
        with pytest.raises(ValueError, match="no method 'tolower'"):
            _eval(df, _eval_args(formula=["y = x.tolower()"]))


@pytest.fixture
def empty_df():
    # An empty TSV is read with float64 columns: the types are unknown
    return pd.DataFrame(
        {"a": pd.Series([], dtype=float), "name": pd.Series([], dtype=float)}
    )


class TestFormulaMistakes:
    def test_no_assignment(self, num_df):
        with pytest.raises(ValueError, match="no assignment.*'name = x \\+ y'"):
            _eval(num_df, _eval_args(formula=["x + y"]))

    def test_special_function_with_unknown_column(self, num_df):
        with pytest.raises(ValueError, match=r"column\(s\) not found: \['x y'\]"):
            _eval(num_df, _eval_args(formula=["z = sum(x y)"]))

    def test_special_function_with_number_suggests_constant_column(self):
        df = pd.DataFrame({"k": [3], "n": [10]})
        with pytest.raises(ValueError, match="-c 'p = 0.5'") as exc:
            _eval(df, _eval_args(formula=["z = binom_test(k, n, 0.5)"]))
        assert "pandas df.eval" not in str(exc.value)

    def test_constant_column_works_with_special_function(self):
        df = pd.DataFrame({"k": [3], "n": [10]})
        args = _eval_args(constant=["p = 0.5"], formula=["z = binom_test(k, n, p)"])
        assert _eval(df, args)["z"].iloc[0] == pytest.approx(0.34375)

    def test_numeric_literal_is_not_a_method_call(self, num_df):
        result = _eval(num_df, _eval_args(formula=["z = 1.5 * x"]))
        assert list(result["z"]) == pytest.approx([1.5, 3.0, 4.5])


class TestEmptyTableFormulas:
    @pytest.mark.parametrize(
        "formula, message",
        [
            ("c = nope + 1", "'nope' is not defined"),
            ("c = a.nosuch()", "no method 'nosuch'"),
            ("c = frobnicate(a)", "unknown function 'frobnicate'"),
            ("c = sum(a, zz)", r"not found: \['zz'\]"),
            ("a + 1", "no assignment"),
        ],
    )
    def test_mistakes_are_reported(self, empty_df, formula, message):
        with pytest.raises(ValueError, match=message):
            _eval(empty_df, _eval_args(formula=[formula]))

    @pytest.mark.parametrize(
        "formula",
        ["c = name.str.lower()", "c = a + 1", "c = sign(a)", "c = basename(name)"],
    )
    def test_type_dependent_formulas_add_the_column(self, empty_df, formula):
        result = _eval(empty_df, _eval_args(formula=[formula]))
        assert "c" in result.columns
        assert len(result) == 0


class TestFormulaDispatch:
    def test_special_function_inside_expression_is_an_error(self, num_df):
        # The old fallback parsed only sum(x, y) and dropped "* 2" silently
        with pytest.raises(ValueError, match=r"sum\(\) must be the whole right-hand"):
            _eval(num_df, _eval_args(formula=["z = sum(x, y) * 2"]))

    def test_special_function_then_expression_in_two_steps(self, num_df):
        args = _eval_args(formula=["tmp = sum(x, y)", "z = tmp * 2"])
        assert list(_eval(num_df, args)["z"]) == pytest.approx([22.0, 44.0, 66.0])

    def test_pandas_math_functions_stay_with_pandas(self, num_df):
        result = _eval(num_df, _eval_args(formula=["z = log1p(x) + expm1(x)"]))
        expected = np.log1p(num_df["x"]) + np.expm1(num_df["x"])
        assert list(result["z"]) == pytest.approx(list(expected))

    def test_syntax_error_is_reported_by_pandas(self, num_df):
        with pytest.raises(ValueError, match="'z = x \\+': invalid syntax") as exc:
            _eval(num_df, _eval_args(formula=["z = x +"]))
        assert "dest = func" not in str(exc.value)
