"""
Tests for dftk.common.docformats and the output formats io.printdf routes to it.
"""

import json

import numpy as np
import pandas as pd
import pytest

from dftk.common import docformats
from dftk.common.io import io
from tests.conftest import make_args


def _out(df, capsys, **overrides):
    """Run io.printdf on *df* and return what it wrote to stdout."""
    io.printdf(df, make_args(**overrides))
    return capsys.readouterr().out


@pytest.fixture
def small():
    return pd.DataFrame(
        {"day": ["Thur", "Fri"], "mean_tip": [2.771452, np.nan], "n": [62, 19]}
    )


@pytest.fixture
def nested():
    """Lists and a struct, as DuckDB hands them to pandas."""
    return pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "tags": [
                np.array(["x", "y"], dtype=object),
                None,
                np.array([], dtype=object),
            ],
            "pos": [{"chr": "1", "start": 100}, None, {"chr": "2", "start": 5}],
        }
    )


@pytest.fixture
def mixed():
    """A DuckDB JSON column: a string in one record, a list in another."""
    df = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "region": ['"chr1:1-2"', '["chr3:1-2","chr3:2-3"]', None],
        }
    )
    docformats.mark_json_columns(df, ["region"])
    return df


class TestNestedAsText:
    def test_lists_and_structs_become_one_line_json(self, nested):
        out = docformats.nested_as_text(nested)
        assert out["tags"].tolist()[0] == '["x","y"]'
        assert out["tags"].tolist()[2] == "[]"
        assert out["pos"].tolist()[0] == '{"chr":"1","start":100}'
        assert out["tags"].isna().tolist() == [False, True, False]

    def test_long_list_stays_on_one_line(self):
        # numpy wraps a long array's str() over several lines: no TSV possible
        regions = np.array([f"chr1:{i}-{i + 500}" for i in range(75)], dtype=object)
        df = pd.DataFrame({"id": ["C9ORF72"], "off": [regions]})
        text = docformats.nested_as_text(df)["off"].iloc[0]
        assert "\n" not in text
        assert json.loads(text) == regions.tolist()

    def test_masked_array_and_decimal(self):
        import decimal

        arr = np.ma.masked_array(
            [decimal.Decimal("1.5"), decimal.Decimal("0")], mask=[False, True]
        )
        df = pd.DataFrame({"l": [arr]})
        assert docformats.nested_as_text(df)["l"].iloc[0] == "[1.5,null]"

    def test_json_column_strings_lose_their_quotes(self, mixed):
        out = docformats.nested_as_text(mixed)
        assert out["region"].tolist()[:2] == ["chr1:1-2", '["chr3:1-2","chr3:2-3"]']

    def test_nan_in_a_list_is_null(self):
        df = pd.DataFrame({"l": [np.array([1.0, np.nan, np.inf])]})
        assert docformats.nested_as_text(df)["l"].iloc[0] == "[1.0,null,null]"

    def test_unchanged_frame_is_returned_as_is(self, small):
        assert docformats.nested_as_text(small) is small


class TestTsvNested:
    def test_select_star_with_lists_prints(self, nested, capsys):
        lines = _out(nested, capsys).splitlines()
        assert lines == [
            "id\ttags\tpos",
            'a\t["x","y"]\t{"chr":"1","start":100}',
            "b\t\t",
            'c\t[]\t{"chr":"2","start":5}',
        ]


class TestRepeatedNames:
    @pytest.fixture
    def twice(self, nested):
        return nested.set_axis(["id", "tags", "id"], axis=1)

    def test_tsv(self, twice, capsys):
        assert _out(twice, capsys).splitlines()[:2] == [
            "id\ttags\tid",
            'a\t["x","y"]\t{"chr":"1","start":100}',
        ]

    def test_markdown(self, twice, capsys):
        assert _out(twice, capsys, format="md").splitlines()[0] == "| id | tags | id |"

    def test_json_is_an_error(self, twice):
        with pytest.raises(ValueError, match="repeated: \\['id'\\]"):
            io.printdf(twice, make_args(format="json"))


class TestJson:
    def test_array_of_records(self, small, capsys):
        records = json.loads(_out(small, capsys, format="json"))
        assert records == [
            {"day": "Thur", "mean_tip": 2.771452, "n": 62},
            {"day": "Fri", "mean_tip": None, "n": 19},
        ]

    def test_json_lines(self, small, capsys):
        lines = _out(small, capsys, format="jsonl").splitlines()
        assert [json.loads(line)["n"] for line in lines] == [62, 19]

    def test_digits(self, small, capsys):
        records = json.loads(_out(small, capsys, format="json", digits=3))
        assert records[0]["mean_tip"] == 2.77

    def test_nested_values_stay_nested(self, nested, capsys):
        records = json.loads(_out(nested, capsys, format="json"))
        assert records[0]["tags"] == ["x", "y"]
        assert records[0]["pos"] == {"chr": "1", "start": 100}
        assert records[1] == {"id": "b", "tags": None, "pos": None}

    def test_json_column_round_trips(self, mixed, capsys):
        records = json.loads(_out(mixed, capsys, format="json"))
        assert [r["region"] for r in records] == [
            "chr1:1-2",
            ["chr3:1-2", "chr3:2-3"],
            None,
        ]

    def test_json_columns_survive_a_parquet_pipe(self, mixed, tmp_path, capsys):
        path = str(tmp_path / "x.parquet")
        io.printdf(mixed, make_args(output=path))
        capsys.readouterr()
        df = io.read(make_args(DATAFILE=path))
        records = json.loads(_out(df, capsys, format="json"))
        assert records[1]["region"] == ["chr3:1-2", "chr3:2-3"]

    def test_move_renames_a_json_column(self, mixed, capsys):
        records = json.loads(_out(mixed, capsys, format="json", move=["region:loc"]))
        assert records[1]["loc"] == ["chr3:1-2", "chr3:2-3"]

    def test_nan_in_a_list_is_null(self, capsys):
        df = pd.DataFrame({"l": [np.array([1.0, np.nan])]})
        assert json.loads(_out(df, capsys, format="json")) == [{"l": [1.0, None]}]

    def test_int64_with_missing_and_bool(self, capsys):
        df = pd.DataFrame(
            {"n": pd.array([1, None], dtype="Int64"), "ok": [True, False]}
        )
        assert json.loads(_out(df, capsys, format="json")) == [
            {"n": 1, "ok": True},
            {"n": None, "ok": False},
        ]


class TestMarkdown:
    def test_pipe_table_numbers_right_aligned(self, small, capsys):
        assert _out(small, capsys, format="md", digits=3).splitlines() == [
            "| day | mean_tip | n |",
            "| --- | ---: | ---: |",
            "| Thur | 2.77 | 62 |",
            "| Fri |  | 19 |",
        ]

    def test_pipe_in_value_is_escaped(self, capsys):
        df = pd.DataFrame({"a": ["x|y"]})
        assert _out(df, capsys, format="md").splitlines()[2] == "| x\\|y |"

    def test_removeheader_leaves_header_empty(self, capsys):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert _out(df, capsys, format="md", removeheader=True).splitlines() == [
            "|  |  |",
            "| ---: | ---: |",
            "| 1 | 2 |",
        ]


class TestLatex:
    def test_bare_booktabs_tabular(self, small, capsys):
        assert _out(small, capsys, format="tex", digits=3).splitlines() == [
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "day & mean\\_tip & n \\\\",
            "\\midrule",
            "Thur & 2.77 & 62 \\\\",
            "Fri &  & 19 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]

    def test_special_characters_escaped(self, capsys):
        df = pd.DataFrame({"a": ["50% & $x$ #1 {a_b} ~ ^ \\"]})
        row = _out(df, capsys, format="tex").splitlines()[4]
        assert row == (
            "50\\% \\& \\$x\\$ \\#1 \\{a\\_b\\} \\textasciitilde{} "
            "\\textasciicircum{} \\textbackslash{} \\\\"
        )

    def test_removeheader_drops_header_and_midrule(self, capsys):
        df = pd.DataFrame({"a": [1]})
        assert _out(df, capsys, format="tex", removeheader=True).splitlines() == [
            "\\begin{tabular}{r}",
            "\\toprule",
            "1 \\\\",
            "\\bottomrule",
            "\\end{tabular}",
        ]

    def test_bar_and_angle_brackets_escaped(self, capsys):
        # The default OT1 font would print an em dash and inverted ! and ?
        df = pd.DataFrame({"a": ["a|b <x>"]})
        row = _out(df, capsys, format="tex").splitlines()[4]
        assert row == "a\\textbar{}b \\textless{}x\\textgreater{} \\\\"

    def test_brackets_braced(self, capsys):
        # "\\" followed by "[" would be read as "\\[length]"
        df = pd.DataFrame({"a": ["[1]"]})
        assert _out(df, capsys, format="tex").splitlines()[4] == "{[}1{]} \\\\"

    def test_bool_column_is_left_aligned(self, capsys):
        df = pd.DataFrame({"ok": [True]})
        assert _out(df, capsys, format="tex").splitlines()[0] == "\\begin{tabular}{l}"


class TestRouting:
    @pytest.mark.parametrize(
        "name, first_line",
        [
            ("t.json", "["),
            ("t.jsonl", '{"day": "Thur", "mean_tip": 2.771452, "n": 62}'),
            ("t.md", "| day | mean_tip | n |"),
            ("t.TEX", "\\begin{tabular}{lrr}"),
            ("t.tsv", "day\tmean_tip\tn"),
        ],
    )
    def test_extension_picks_format(self, small, tmp_path, capsys, name, first_line):
        path = tmp_path / name
        assert _out(small, capsys, output=str(path)) == ""
        assert path.read_text().splitlines()[0] == first_line

    def test_format_overrides_extension(self, small, tmp_path, capsys):
        path = tmp_path / "t.json"
        _out(small, capsys, output=str(path), format="tsv")
        assert path.read_text().startswith("day\tmean_tip\tn\n")

    @pytest.mark.parametrize("output", ["", "x.parquet"])
    def test_format_with_parquet_output_is_an_error(self, small, output):
        with pytest.raises(ValueError, match="can't be combined with parquet"):
            io.printdf(small, make_args(output=output, format="md"))
