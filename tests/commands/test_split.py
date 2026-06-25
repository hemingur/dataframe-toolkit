"""Tests for stattools.commands.split_cmd._split and _make_filename."""

import os

import pandas as pd
import pytest

from stattools.commands.split_cmd import _make_filename, _split
from tests.conftest import make_args


def _args(**kwargs):
    defaults = dict(
        groups=["g"],
        prefix="split-",
        suffix="",
        template=None,
        noclobber=False,
        separator="\t",
        output=None,
        noheader=False,
    )
    defaults.update(kwargs)
    return make_args(**defaults)


@pytest.fixture
def simple_df():
    return pd.DataFrame({"g": ["A", "A", "B", "B"], "x": [1.0, 2.0, 3.0, 4.0]})


@pytest.fixture
def multi_group_df():
    return pd.DataFrame(
        {
            "g1": ["X", "X", "Y", "Y"],
            "g2": ["P", "Q", "P", "Q"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )


# ---------------------------------------------------------------------------
# _make_filename
# ---------------------------------------------------------------------------


class TestMakeFilename:
    def test_simple_string_group(self):
        assert _make_filename("foo", "out/", ".tsv") == "out/foo.tsv"

    def test_tuple_group(self):
        assert _make_filename(("A", "B"), "split-", ".tsv") == "split-A_B.tsv"

    def test_numeric_group(self):
        assert _make_filename(42, "f-", "") == "f-42"

    def test_default_prefix_suffix(self):
        assert _make_filename("bar", "split-", "") == "split-bar"


# ---------------------------------------------------------------------------
# _split — file creation
# ---------------------------------------------------------------------------


class TestSplitFiles:
    def test_creates_one_file_per_group(self, tmp_path, simple_df):
        _split(simple_df, _args(prefix=str(tmp_path) + "/", suffix=".tsv"))
        files = sorted(os.listdir(tmp_path))
        assert files == ["A.tsv", "B.tsv"]

    def test_file_contents_correct(self, tmp_path, simple_df):
        _split(simple_df, _args(prefix=str(tmp_path) + "/", suffix=".tsv"))
        df_a = pd.read_csv(tmp_path / "A.tsv", sep="\t")
        assert list(df_a["g"]) == ["A", "A"]
        assert list(df_a["x"]) == [1.0, 2.0]

    def test_multi_group_key_filename(self, tmp_path, multi_group_df):
        _split(
            multi_group_df,
            _args(groups=["g1", "g2"], prefix=str(tmp_path) + "/", suffix=".tsv"),
        )
        files = sorted(os.listdir(tmp_path))
        assert files == ["X_P.tsv", "X_Q.tsv", "Y_P.tsv", "Y_Q.tsv"]

    def test_template_filename(self, tmp_path, simple_df):
        template = str(tmp_path / "data_{g}.tsv")
        _split(simple_df, _args(template=template))
        assert (tmp_path / "data_A.tsv").exists()
        assert (tmp_path / "data_B.tsv").exists()

    def test_template_multi_group(self, tmp_path, multi_group_df):
        template = str(tmp_path / "{g1}_{g2}.tsv")
        _split(multi_group_df, _args(groups=["g1", "g2"], template=template))
        assert (tmp_path / "X_P.tsv").exists()

    def test_template_unknown_placeholder_preserved(self, tmp_path, simple_df):
        template = str(tmp_path / "{g}_{unknown}.tsv")
        _split(simple_df, _args(template=template))
        assert (tmp_path / "A_{unknown}.tsv").exists()

    def test_noclobber_skips_existing(self, tmp_path, simple_df):
        path = tmp_path / "A.tsv"
        path.write_text("original\n")
        _split(
            simple_df, _args(prefix=str(tmp_path) + "/", suffix=".tsv", noclobber=True)
        )
        assert path.read_text() == "original\n"

    def test_no_noclobber_overwrites(self, tmp_path, simple_df):
        path = tmp_path / "A.tsv"
        path.write_text("original\n")
        _split(
            simple_df, _args(prefix=str(tmp_path) + "/", suffix=".tsv", noclobber=False)
        )
        assert path.read_text() != "original\n"

    def test_group_col_preserved_in_output(self, tmp_path, simple_df):
        _split(simple_df, _args(prefix=str(tmp_path) + "/", suffix=".tsv"))
        df_b = pd.read_csv(tmp_path / "B.tsv", sep="\t")
        assert "g" in df_b.columns
