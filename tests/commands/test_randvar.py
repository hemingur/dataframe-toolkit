"""
Tests for stattools.commands.randvar_cmd.
"""

import pandas as pd
import pytest

from stattools.commands.randvar_cmd import (
    RandvarCommand,
    _list_distributions,
    _parse_parameters,
)
from stattools.common.seed import normalize_seed
from tests.conftest import make_args


def _make_args(**kwargs):
    defaults = dict(
        destcol="x",
        dist="norm",
        parameters=None,
        randomseed=None,
        nsamples=None,
        list=False,
    )
    defaults.update(kwargs)
    return make_args(**defaults)


# ---------------------------------------------------------------------------
# normalize_seed (common.seed)
# ---------------------------------------------------------------------------


def test_resolve_seed_none():
    assert normalize_seed(None) is None


def test_resolve_seed_int_string():
    assert normalize_seed("42") == 42


def test_resolve_seed_negative_int():
    assert normalize_seed("-1") == -1


def test_resolve_seed_string_hashed():
    seed = normalize_seed("hello")
    assert isinstance(seed, int)
    assert 0 <= seed < (1 << 32)


def test_resolve_seed_string_reproducible():
    assert normalize_seed("myseed") == normalize_seed("myseed")


def test_resolve_seed_different_strings_differ():
    assert normalize_seed("aaa") != normalize_seed("bbb")


# ---------------------------------------------------------------------------
# _parse_parameters
# ---------------------------------------------------------------------------


def test_parse_parameters_none():
    assert _parse_parameters(None) == {}


def test_parse_parameters_empty():
    assert _parse_parameters("") == {}


def test_parse_parameters_single():
    assert _parse_parameters("loc:5") == {"loc": 5.0}


def test_parse_parameters_multiple():
    result = _parse_parameters("loc:0,scale:2")
    assert result == {"loc": 0.0, "scale": 2.0}


def test_parse_parameters_shape_params():
    result = _parse_parameters("a:2,b:5,loc:0,scale:1")
    assert result == {"a": 2.0, "b": 5.0, "loc": 0.0, "scale": 1.0}


# ---------------------------------------------------------------------------
# _list_distributions
# ---------------------------------------------------------------------------


def test_list_distributions_columns():
    df = _list_distributions()
    assert set(df.columns) == {"name", "type", "parameters"}


def test_list_distributions_has_continuous():
    df = _list_distributions()
    cont = df[df["type"] == "continuous"]
    assert "norm" in cont["name"].values
    assert "beta" in cont["name"].values
    assert "uniform" in cont["name"].values


def test_list_distributions_has_discrete():
    df = _list_distributions()
    disc = df[df["type"] == "discrete"]
    assert "poisson" in disc["name"].values
    assert "binom" in disc["name"].values


def test_list_distributions_norm_params():
    df = _list_distributions()
    row = df[df["name"] == "norm"].iloc[0]
    assert "loc" in row["parameters"]
    assert "scale" in row["parameters"]


def test_list_distributions_beta_params():
    df = _list_distributions()
    row = df[df["name"] == "beta"].iloc[0]
    assert "a" in row["parameters"]
    assert "b" in row["parameters"]


def test_list_distributions_poisson_no_scale():
    df = _list_distributions()
    row = df[df["name"] == "poisson"].iloc[0]
    assert "scale" not in row["parameters"]


# ---------------------------------------------------------------------------
# RandvarCommand.execute — standalone (-n) mode
# ---------------------------------------------------------------------------


def test_standalone_norm(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=100, dist="norm", randomseed="42")
    cmd.execute(args)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert lines[0] == "x"
    assert len(lines) == 101  # header + 100 rows


def test_standalone_correct_count(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=50, dist="uniform", randomseed="0")
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert len(df) == 50
    assert "x" in df.columns


def test_standalone_reproducible(capsys):
    cmd = RandvarCommand()
    cmd.execute(_make_args(nsamples=20, dist="norm", randomseed="99"))
    out1 = capsys.readouterr().out
    cmd.execute(_make_args(nsamples=20, dist="norm", randomseed="99"))
    out2 = capsys.readouterr().out
    assert out1 == out2


def test_standalone_different_seeds_differ(capsys):
    cmd = RandvarCommand()
    cmd.execute(_make_args(nsamples=50, dist="norm", randomseed="1"))
    out1 = capsys.readouterr().out
    cmd.execute(_make_args(nsamples=50, dist="norm", randomseed="2"))
    out2 = capsys.readouterr().out
    assert out1 != out2


def test_standalone_with_parameters(capsys):
    cmd = RandvarCommand()
    args = _make_args(
        nsamples=1000, dist="norm", parameters="loc:100,scale:0.001", randomseed="7"
    )
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert df["x"].mean() == pytest.approx(100.0, abs=0.1)


def test_standalone_discrete_poisson(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=500, dist="poisson", parameters="mu:3", randomseed="5")
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert len(df) == 500
    assert (df["x"] >= 0).all()


def test_standalone_custom_destcol(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=10, dist="uniform", destcol="myrand", randomseed="0")
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert "myrand" in df.columns


# ---------------------------------------------------------------------------
# RandvarCommand.execute — append-to-dataframe mode
# ---------------------------------------------------------------------------


def _run_on_df(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Run RandvarCommand on *df* via monkeypatched io.read."""
    import stattools.commands.randvar_cmd as mod

    original_read = mod.io.read
    mod.io.read = lambda args: df.copy()
    try:
        cmd = RandvarCommand()
        args = _make_args(DATAFILE="fake.tsv", **kwargs)
        import io as _io
        from contextlib import redirect_stdout

        buf = _io.StringIO()
        with redirect_stdout(buf):
            cmd.execute(args)
        return pd.read_csv(_io.StringIO(buf.getvalue()), sep="\t")
    finally:
        mod.io.read = original_read


def test_append_column_present():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = _run_on_df(df, dist="norm", randomseed="42")
    assert "a" in result.columns
    assert "x" in result.columns


def test_append_column_count_matches():
    df = pd.DataFrame({"a": range(7)})
    result = _run_on_df(df, dist="uniform", randomseed="0")
    assert len(result) == 7


def test_append_preserves_existing_columns():
    df = pd.DataFrame({"a": [10, 20], "b": ["foo", "bar"]})
    result = _run_on_df(df, dist="norm", randomseed="1")
    assert list(result["a"]) == [10, 20]
    assert list(result["b"]) == ["foo", "bar"]


def test_append_uniform_in_range():
    df = pd.DataFrame({"a": range(200)})
    result = _run_on_df(df, dist="uniform", randomseed="3")
    assert (result["x"] >= 0).all()
    assert (result["x"] <= 1).all()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_bad_distribution_exits(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=10, dist="notadist")
    with pytest.raises(SystemExit):
        cmd.execute(args)


def test_bad_parameters_exits(capsys):
    cmd = RandvarCommand()
    args = _make_args(nsamples=10, dist="norm", parameters="mean:0,var:1")
    with pytest.raises(SystemExit):
        cmd.execute(args)


def test_missing_destcol_exits():
    cmd = RandvarCommand()
    args = _make_args(nsamples=10, dist="norm", destcol=None)
    with pytest.raises(SystemExit):
        cmd.execute(args)


def test_missing_dist_exits():
    cmd = RandvarCommand()
    args = _make_args(nsamples=10, dist=None)
    with pytest.raises(SystemExit):
        cmd.execute(args)


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def test_list_flag_prints_tsv(capsys):
    cmd = RandvarCommand()
    args = _make_args(list=True)
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert set(df.columns) == {"name", "type", "parameters"}
    assert len(df) > 100


def test_list_flag_bypasses_destcol_requirement(capsys):
    """--list must succeed even when destcol and dist are None."""
    cmd = RandvarCommand()
    args = _make_args(list=True, destcol=None, dist=None)
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert "name" in df.columns


def test_list_distributions_sorted():
    """Distribution names within each type should be sorted."""
    df = _list_distributions()
    cont = df[df["type"] == "continuous"]["name"].tolist()
    disc = df[df["type"] == "discrete"]["name"].tolist()
    assert cont == sorted(cont)
    assert disc == sorted(disc)


def test_list_distributions_both_types_present():
    """Both continuous and discrete types must appear in the listing."""
    df = _list_distributions()
    assert set(df["type"].unique()) == {"continuous", "discrete"}


# ---------------------------------------------------------------------------
# _parse_parameters — additional edge cases
# ---------------------------------------------------------------------------


def test_parse_parameters_negative_float():
    result = _parse_parameters("loc:-3.5,scale:1.0")
    assert result == {"loc": -3.5, "scale": 1.0}


def test_parse_parameters_fractional_shape():
    result = _parse_parameters("a:0.5,b:2.0")
    assert result["a"] == pytest.approx(0.5)
    assert result["b"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# RandvarCommand.execute — append mode, additional distributions
# ---------------------------------------------------------------------------


def test_append_beta_in_unit_interval():
    """Beta samples must lie in [0, 1]."""
    df = pd.DataFrame({"a": range(200)})
    result = _run_on_df(df, dist="beta", parameters="a:2,b:5", randomseed="10")
    assert (result["x"] >= 0).all()
    assert (result["x"] <= 1).all()


def test_append_discrete_poisson_nonnegative():
    """Poisson samples appended to an existing frame must be non-negative integers."""
    df = pd.DataFrame({"a": range(50)})
    result = _run_on_df(df, dist="poisson", parameters="mu:4", randomseed="20")
    assert len(result) == 50
    assert (result["x"] >= 0).all()
    # Values should all be integers (no fractional parts)
    assert (result["x"] == result["x"].round()).all()


def test_append_custom_destcol_on_existing_df():
    """Append mode respects --destcol even when reading a real DataFrame."""
    df = pd.DataFrame({"score": [0.1, 0.5, 0.9]})
    result = _run_on_df(df, dist="uniform", destcol="noise", randomseed="0")
    assert "score" in result.columns
    assert "noise" in result.columns
    assert len(result) == 3


def test_standalone_beta_in_unit_interval(capsys):
    """Standalone beta samples must be in [0, 1]."""
    cmd = RandvarCommand()
    args = _make_args(nsamples=500, dist="beta", parameters="a:2,b:3", randomseed="77")
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert (df["x"] >= 0).all()
    assert (df["x"] <= 1).all()


def test_standalone_uniform_mean_approx_half(capsys):
    """Large uniform sample should have mean close to 0.5."""
    cmd = RandvarCommand()
    args = _make_args(nsamples=2000, dist="uniform", randomseed="123")
    cmd.execute(args)
    out = capsys.readouterr().out
    df = pd.read_csv(__import__("io").StringIO(out), sep="\t")
    assert df["x"].mean() == pytest.approx(0.5, abs=0.05)
