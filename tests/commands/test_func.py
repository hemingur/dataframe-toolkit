"""Tests for dftk.commands.func_cmd."""

import argparse
import io as _io
import os
import sys
import tempfile

import pandas as pd
import pytest

from dftk.commands.func_cmd import FuncCommand, _apply_transform


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "value": [1.0, 3.0, 2.0, 4.0, 6.0],
            "label": ["x", "y", "x", "y", "z"],
        }
    )


def _run(df, **kwargs):
    with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
        df.to_csv(f, sep="\t", index=False)
        fname = f.name
    defaults = dict(
        DATAFILE=fname,
        groupcol=None,
        destcol=None,
        xcol=None,
        select=None,
        drop=None,
        move=None,
        na_rep=None,
        dropna=False,
        postquery=[],
        cast=None,
        sortasc=None,
        sortdesc=None,
        sort=None,
        round=None,
        deduplicate=None,
        noheader=False,
        removeheader=False,
        output=None,
        digits=None,
        errortag="-",
        backend="pandas",
        delimiter=None,
        readasobject=None,
        prequery=[],
    )
    defaults.update(kwargs)
    args = argparse.Namespace(**defaults)
    buf = _io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        FuncCommand().execute(args)
    finally:
        sys.stdout = old
        os.unlink(fname)
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    header = lines[0].split("\t")
    rows = [dict(zip(header, line.split("\t"), strict=False)) for line in lines[1:]]
    return rows


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------


class TestCumsum:
    def test_cumsum_produces_column(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert "value_cumsum" in rows[0]

    def test_cumsum_values(self, df):
        rows = _run(df, col="value", transform="cumsum")
        expected = [1.0, 4.0, 6.0, 10.0, 16.0]
        actual = [float(r["value_cumsum"]) for r in rows]
        assert actual == pytest.approx(expected)

    def test_cumsum_grouped(self, df):
        rows = _run(df, col="value", transform="cumsum", groupcol=["group"])
        # A: 1, 4  B: 2, 6, 12
        expected = {"A": [1.0, 4.0], "B": [2.0, 6.0, 12.0]}
        by_group: dict[str, list[float]] = {}
        for r in rows:
            by_group.setdefault(r["group"], []).append(float(r["value_cumsum"]))
        assert by_group == pytest.approx(expected)

    def test_cumsum_custom_destcol(self, df):
        rows = _run(df, col="value", transform="cumsum", destcol="running_total")
        assert "running_total" in rows[0]

    def test_cumsum_row_count_preserved(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert len(rows) == len(df)


# ---------------------------------------------------------------------------
# Group aggregates (mean, sum, min, max, count, median, std)
# ---------------------------------------------------------------------------


class TestGroupAggregates:
    def test_group_mean(self, df):
        rows = _run(df, col="value", transform="mean", groupcol=["group"])
        # A mean = 2.0, B mean = 4.0
        a_rows = [float(r["value_mean"]) for r in rows if r["group"] == "A"]
        b_rows = [float(r["value_mean"]) for r in rows if r["group"] == "B"]
        assert all(v == pytest.approx(2.0) for v in a_rows)
        assert all(v == pytest.approx(4.0) for v in b_rows)

    def test_group_sum(self, df):
        rows = _run(df, col="value", transform="sum", groupcol=["group"])
        a_rows = [float(r["value_sum"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(4.0) for v in a_rows)

    def test_group_min(self, df):
        rows = _run(df, col="value", transform="min", groupcol=["group"])
        a_rows = [float(r["value_min"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(1.0) for v in a_rows)

    def test_group_max(self, df):
        rows = _run(df, col="value", transform="max", groupcol=["group"])
        b_rows = [float(r["value_max"]) for r in rows if r["group"] == "B"]
        assert all(v == pytest.approx(6.0) for v in b_rows)

    def test_group_count(self, df):
        rows = _run(df, col="value", transform="count", groupcol=["group"])
        a_rows = [float(r["value_count"]) for r in rows if r["group"] == "A"]
        assert all(v == pytest.approx(2.0) for v in a_rows)

    def test_global_mean_no_group(self, df):
        rows = _run(df, col="value", transform="mean")
        # global mean of [1, 3, 2, 4, 6] = 3.2
        means = [float(r["value_mean"]) for r in rows]
        assert all(m == pytest.approx(3.2) for m in means)

    def test_row_count_preserved(self, df):
        rows = _run(df, col="value", transform="mean", groupcol=["group"])
        assert len(rows) == len(df)


# ---------------------------------------------------------------------------
# rank / pct_rank
# ---------------------------------------------------------------------------


class TestRank:
    def test_rank_produces_column(self, df):
        rows = _run(df, col="value", transform="rank")
        assert "value_rank" in rows[0]

    def test_rank_grouped(self, df):
        rows = _run(df, col="value", transform="rank", groupcol=["group"])
        # A: values 1,3 → ranks 1,2  B: values 2,4,6 → ranks 1,2,3
        a_ranks = [float(r["value_rank"]) for r in rows if r["group"] == "A"]
        b_ranks = [float(r["value_rank"]) for r in rows if r["group"] == "B"]
        assert sorted(a_ranks) == pytest.approx([1.0, 2.0])
        assert sorted(b_ranks) == pytest.approx([1.0, 2.0, 3.0])

    def test_pct_rank_range(self, df):
        rows = _run(df, col="value", transform="pct_rank")
        pcts = [float(r["value_pct_rank"]) for r in rows]
        assert all(0.0 < p <= 1.0 for p in pcts)


# ---------------------------------------------------------------------------
# qcut
# ---------------------------------------------------------------------------


class TestQcut:
    def test_qcut_produces_column(self, df):
        rows = _run(df, col="value", transform="qcut:2")
        assert "value_qcut_2" in rows[0]

    def test_qcut_bins_in_range(self, df):
        rows = _run(df, col="value", transform="qcut:2")
        bins = {int(r["value_qcut_2"]) for r in rows}
        assert bins.issubset({1, 2})

    def test_qcut_grouped(self, df):
        rows = _run(df, col="value", transform="qcut:2", groupcol=["group"])
        # Each group should only have bins 1 or 2
        bins = {int(r["value_qcut_2"]) for r in rows}
        assert bins.issubset({1, 2})

    def test_qcut_invalid_n(self, df):
        import os as _os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                col="value",
                transform="qcut:1",
                groupcol=None,
                destcol=None,
                select=None,
                drop=None,
                move=None,
                na_rep=None,
                dropna=False,
                postquery=[],
                cast=None,
                sortasc=None,
                sortdesc=None,
                sort=None,
                round=None,
                deduplicate=None,
                noheader=False,
                removeheader=False,
                output=None,
                digits=None,
                errortag="-",
                backend="pandas",
                delimiter=None,
                readasobject=None,
                prequery=[],
            )
            with pytest.raises(ValueError, match="qcut requires N >= 2"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_col_raises(self, df):
        import os as _os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                col="no_such_col",
                transform="cumsum",
                groupcol=None,
                destcol=None,
                select=None,
                drop=None,
                move=None,
                na_rep=None,
                dropna=False,
                postquery=[],
                cast=None,
                sortasc=None,
                sortdesc=None,
                sort=None,
                round=None,
                deduplicate=None,
                noheader=False,
                removeheader=False,
                output=None,
                digits=None,
                errortag="-",
                backend="pandas",
                delimiter=None,
                readasobject=None,
                prequery=[],
            )
            with pytest.raises(ValueError, match="no_such_col"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)

    def test_unknown_transform_raises(self, df):
        import os as _os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".tsv", mode="w", delete=False) as f:
            df.to_csv(f, sep="\t", index=False)
            fname = f.name
        try:
            args = argparse.Namespace(
                DATAFILE=fname,
                col="value",
                transform="bogus",
                groupcol=None,
                destcol=None,
                select=None,
                drop=None,
                move=None,
                na_rep=None,
                dropna=False,
                postquery=[],
                cast=None,
                sortasc=None,
                sortdesc=None,
                sort=None,
                round=None,
                deduplicate=None,
                noheader=False,
                removeheader=False,
                output=None,
                digits=None,
                errortag="-",
                backend="pandas",
                delimiter=None,
                readasobject=None,
                prequery=[],
            )
            with pytest.raises(ValueError, match="Unknown transform"):
                FuncCommand().execute(args)
        finally:
            _os.unlink(fname)

    def test_default_destcol_name(self, df):
        rows = _run(df, col="value", transform="cumsum")
        assert "value_cumsum" in rows[0]

    def test_default_destcol_qcut(self, df):
        rows = _run(df, col="value", transform="qcut:4")
        assert "value_qcut_4" in rows[0]


# ---------------------------------------------------------------------------
# interpolate / ffill / bfill (issue #15)
# ---------------------------------------------------------------------------

nan = float("nan")


@pytest.fixture
def sites():
    """Two sites, unevenly spaced days, rows out of order, gaps at both ends."""
    return pd.DataFrame(
        {
            "site": ["B", "A", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "day": [4, 6, 1, 10, 8, 0, 2, 5, 12, 3, 7],
            "temp": [nan, 20, 10, 15, 24, nan, nan, nan, nan, 5, nan],
        }
    )


def _fill(df, transform="interpolate", groupcols=("site",), xcol="day"):
    out = _apply_transform(
        df.copy(), "temp", transform, list(groupcols) if groupcols else None,
        "out", xcol,
    )  # fmt: skip
    return out["out"].tolist()


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        if pd.isna(y):
            assert pd.isna(x)
        else:
            assert x == pytest.approx(y)


class TestInterpolate:
    def test_by_x_within_groups_in_file_order(self, sites):
        # A: day 2 between 1 (10) and 6 (20) -> 12; day 7 -> 22
        # B: days 4, 5 between 3 (5) and 10 (15) -> 5 + 10/7, 5 + 20/7
        _same(
            _fill(sites),
            [5 + 10 / 7, 20, 10, 15, 24, nan, 12, 5 + 20 / 7, nan, 5, 22],
        )

    def test_known_values_unchanged(self, sites):
        out = _fill(sites)
        known = sites["temp"].notna()
        assert [o for o, k in zip(out, known, strict=True) if k] == list(
            sites["temp"][known]
        )

    def test_no_extrapolation(self, sites):
        out = _fill(sites)
        assert pd.isna(out[5])  # B day 0, before B's first value
        assert pd.isna(out[8])  # A day 12, after A's last value

    def test_groups_do_not_mix(self):
        df = pd.DataFrame(
            {"site": ["A", "A", "B", "B"], "day": [1, 2, 3, 4],
             "temp": [1.0, nan, nan, 100.0]}
        )  # fmt: skip
        _same(_fill(df), [1, nan, nan, 100])

    @pytest.mark.parametrize("transform", ["interpolate", "ffill", "bfill"])
    def test_xcol_required(self, sites, transform):
        with pytest.raises(ValueError, match="needs -x/--xcol"):
            _fill(sites, transform, xcol=None)

    def test_without_groups(self):
        df = pd.DataFrame({"day": [0, 1, 4], "temp": [0.0, nan, 8.0]})
        _same(_fill(df, groupcols=None), [0, 2, 8])

    def test_repeated_x_is_an_error_naming_the_group(self, sites):
        df = pd.concat([sites, pd.DataFrame({"site": ["B"], "day": [3], "temp": [6]})])
        with pytest.raises(ValueError, match="site=B"):
            _fill(df)

    def test_rows_without_x_left_alone(self):
        df = pd.DataFrame({"day": [0, nan, 2], "temp": [0.0, nan, 2.0]})
        _same(_fill(df, groupcols=None), [0, nan, 2])

    def test_one_known_value_nothing_to_draw(self):
        df = pd.DataFrame({"day": [0, 1, 2], "temp": [nan, 5.0, nan]})
        _same(_fill(df, groupcols=None), [nan, 5, nan])

    def test_text_column_is_an_error(self):
        df = pd.DataFrame({"day": [0, 1], "temp": ["a", None]})
        with pytest.raises(ValueError, match="numeric"):
            _fill(df, groupcols=None)

    def test_text_xcol_is_an_error(self):
        df = pd.DataFrame({"day": ["a", "b"], "temp": [1.0, nan]})
        with pytest.raises(ValueError, match="--xcol"):
            _fill(df, groupcols=None)

    def test_group_with_missing_key(self):
        df = pd.DataFrame(
            {"site": [None, None, None], "day": [0, 1, 2], "temp": [0.0, nan, 2.0]}
        )
        _same(_fill(df), [0, 1, 2])


class TestFfillBfill:
    def test_ffill_in_x_order(self, sites):
        # A in day order: 1:10, 2:nan, 6:20, 7:nan, 8:24, 12:nan
        # B in day order: 0:nan, 3:5, 4:nan, 5:nan, 10:15
        _same(
            _fill(sites, "ffill"),
            [5, 20, 10, 15, 24, nan, 10, 5, 24, 5, 20],
        )

    def test_bfill_in_x_order(self, sites):
        _same(
            _fill(sites, "bfill"),
            [15, 20, 10, 15, 24, 5, 20, 15, nan, 5, 24],
        )

    def test_ffill_text(self):
        df = pd.DataFrame(
            {"site": ["A", "A", "A"], "day": [3, 1, 2], "temp": [None, "x", None]}
        )
        assert _fill(df, "ffill") == ["x", "x", "x"]

    def test_ties_in_file_order(self):
        df = pd.DataFrame({"day": [1, 1, 2], "temp": ["a", "b", None]})
        assert _fill(df, "ffill", groupcols=None) == ["a", "b", "b"]

    def test_chain_leaves_no_gaps(self, sites):
        df = sites.copy()
        for t in ("interpolate", "ffill", "bfill"):
            df["temp"] = _fill(df, t)
        # ends take the nearest known value: B day 0 <- 5, A day 12 <- 24
        _same(
            df["temp"].tolist(),
            [5 + 10 / 7, 20, 10, 15, 24, 5, 12, 5 + 20 / 7, 24, 5, 22],
        )


class TestFillTypes:
    """Column and X types, and edge cases from the review of #42."""

    def test_nullable_int_values_and_x(self):
        df = pd.DataFrame(
            {"day": pd.array([0, 1, 4, None], dtype="Int64"),
             "temp": pd.array([0, None, 8, None], dtype="Int64")}
        )  # fmt: skip
        _same(_fill(df, groupcols=None), [0, 2, 8, nan])

    def test_int_column_comes_back_decimal(self):
        df = pd.DataFrame({"day": [0, 1], "temp": [1, 2]})
        out = _apply_transform(df.copy(), "temp", "interpolate", None, "d", "day")
        assert out["d"].dtype == float and out["d"].tolist() == [1.0, 2.0]

    def test_bool_column_is_an_error(self):
        df = pd.DataFrame({"day": [0, 1, 2], "temp": [True, None, False]})
        df["temp"] = df["temp"].astype("boolean")
        with pytest.raises(ValueError, match="numeric"):
            _fill(df, groupcols=None)

    def test_bool_x_is_an_error_for_interpolate(self):
        df = pd.DataFrame({"day": [True, False], "temp": [1.0, nan]})
        with pytest.raises(ValueError, match="true/false"):
            _fill(df, groupcols=None)

    def test_date_x_interpolate(self):
        df = pd.DataFrame(
            {"day": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-11", None]),
             "temp": [0.0, nan, 10.0, nan]}
        )  # fmt: skip
        _same(_fill(df, groupcols=None), [0, 1, 10, nan])

    def test_timezone_date_x_interpolate(self):
        days = pd.to_datetime(["2026-01-01", "2026-01-03", "2026-01-05"])
        df = pd.DataFrame(
            {"day": days.tz_localize("Europe/Oslo"), "temp": [0.0, nan, 4.0]}
        )
        _same(_fill(df, groupcols=None), [0, 2, 4])

    @pytest.mark.parametrize(
        "x",
        [
            pd.to_datetime(["2026-03-01", "2026-01-01", "2026-02-01"]),
            ["2026-03-01", "2026-01-01", "2026-02-01"],  # ISO dates as text
        ],
    )
    def test_ffill_date_or_text_x(self, x):
        df = pd.DataFrame({"day": x, "temp": [nan, 1.0, nan]})
        _same(_fill(df, "ffill", groupcols=None), [1, 1, 1])
        _same(_fill(df, "bfill", groupcols=None), [nan, 1, nan])

    def test_text_x_is_an_error_for_interpolate(self):
        df = pd.DataFrame({"day": ["a", "b"], "temp": [1.0, nan]})
        with pytest.raises(ValueError, match="numbers or dates"):
            _fill(df, groupcols=None)

    def test_ffill_rows_without_x_left_alone(self):
        df = pd.DataFrame({"day": [0, nan, 2], "temp": [1.0, nan, nan]})
        _same(_fill(df, "ffill", groupcols=None), [1, nan, 1])

    def test_bfill_missing_group_key_is_its_own_group(self):
        df = pd.DataFrame(
            {"site": [None, None, "A"], "day": [0, 1, 2], "temp": [nan, 1.0, 9.0]}
        )
        _same(_fill(df, "bfill"), [1, 1, 9])

    def test_two_group_columns_with_a_missing_key(self):
        df = pd.DataFrame(
            {"site": ["A", "A", "A", "A"], "depth": [None, None, 1.0, 1.0],
             "day": [0, 1, 0, 1], "temp": [1.0, nan, nan, 5.0]}
        )  # fmt: skip
        _same(_fill(df, "ffill", groupcols=("site", "depth")), [1, 1, nan, 5])

    def test_repeated_x_error_even_without_gaps(self):
        df = pd.DataFrame({"day": [1, 1], "temp": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Repeated"):
            _fill(df, groupcols=None)

    @pytest.mark.parametrize("transform", ["interpolate", "ffill", "bfill"])
    @pytest.mark.parametrize("groupcols", [None, ("site",)])
    def test_empty_table(self, transform, groupcols):
        df = pd.DataFrame({"site": [], "day": [], "temp": []}, dtype=float)
        assert _fill(df, transform, groupcols=groupcols) == []


class TestFillCommand:
    def test_end_to_end(self, sites):
        rows = _run(
            sites, col="temp", transform="interpolate", groupcol=["site"],
            xcol="day",
        )  # fmt: skip
        assert [r["site"] + r["day"] for r in rows] == [
            f"{s}{d}" for s, d in zip(sites["site"], sites["day"], strict=True)
        ]
        assert float(rows[6]["temp_interpolate"]) == pytest.approx(12.0)
        assert rows[8]["temp_interpolate"] == ""

    def test_xcol_with_other_transform_is_an_error(self, sites):
        with pytest.raises(ValueError, match="--xcol"):
            _run(sites, col="temp", transform="cumsum", xcol="day")

    def test_missing_xcol(self, sites):
        with pytest.raises(ValueError, match="-x/--xcol"):
            _run(sites, col="temp", transform="interpolate", xcol="nosuch")
