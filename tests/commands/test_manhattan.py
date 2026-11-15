"""
Tests for dftk manhattan.  Figures are saved with the Agg backend; the
points drawn are checked through the scatter calls.
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dftk.commands.manhattan_cmd import ManhattanCommand, gwas_table  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    ManhattanCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


@pytest.fixture
def scan():
    return pd.DataFrame(
        {
            "Chr": ["2", "1", "1", "X", "2", "1"],
            "pos": [500, 100, 300, 50, 900, 200],
            "Pval": [1e-10, 0.5, 1e-3, 0.2, 1e-40, np.nan],
            "study": ["a", "a", "b", "b", "a", "b"],
        }
    )


@pytest.fixture
def scatters(monkeypatch):
    """Record (x, y, kwargs) of every Axes.scatter call."""
    calls = []
    orig = matplotlib.axes.Axes.scatter

    def _record(self, x, y, *a, **kw):
        calls.append((np.asarray(x, float), np.asarray(y, float), kw))
        return orig(self, x, y, *a, **kw)

    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", _record)
    return calls


def _run(tmp_path, df, *argv):
    """Run the command to a PNG; return the figure it saved."""
    from unittest.mock import patch

    import dftk.commands.manhattan_cmd as mod

    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = mod.save_or_show
    with patch.object(mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)):
        ManhattanCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0]


class TestGwasTable:
    def test_default_columns_and_missing_rows(self, scan):
        out = gwas_table(scan, _args())
        assert len(out) == 5  # the NaN p-value is left out
        assert out["_y"].tolist() == pytest.approx(
            [10, -np.log10(0.5), 3, -np.log10(0.2), 40]
        )

    def test_named_columns(self, scan):
        df = scan.rename(columns={"Chr": "CHROM", "pos": "BP", "Pval": "p_wald"})
        out = gwas_table(df, _args("--pcol", "p_wald"))
        assert len(out) == 5

    def test_chi2_column(self):
        df = pd.DataFrame({"Chr": ["1"], "pos": [1], "stat": [2000.0], "k": [1]})
        out = gwas_table(df, _args("--chi2col", "stat", "--chi2df", "k"))
        assert out["_y"].iloc[0] == pytest.approx(436.043273716073)

    def test_p_columns_exclusive(self):
        with pytest.raises(SystemExit):
            _args("--pcol", "a", "--chi2col", "b")

    def test_nothing_left_is_an_error(self):
        df = pd.DataFrame({"Chr": ["1"], "pos": [1], "Pval": [np.nan]})
        with pytest.raises(ValueError, match="No rows"):
            gwas_table(df, _args())


class TestThreshold:
    @pytest.mark.parametrize("bad", ["0", "1", "x", "-1e-8"])
    def test_bad_values(self, bad):
        with pytest.raises(SystemExit):
            _args("--sig", bad)

    def test_none(self):
        assert _args("--sig", "none", "--suggestive", "NONE").sig is None


class TestExecute:
    def test_points_in_genome_order(self, tmp_path, scan, scatters):
        _run(tmp_path, scan, "--no-thin")
        x, y, kw = scatters[0]
        # rows in file order; chr1 spans 100..300, chr2 starts after it
        assert x[1] < x[2] < x[0] < x[4] < x[3]
        assert y == pytest.approx([10, -np.log10(0.5), 3, -np.log10(0.2), 40])
        assert kw["rasterized"]
        # alternating colours: chr1 and X dark, chr2 light
        assert kw["c"][1] == kw["c"][3] != kw["c"][0]

    def test_chromosome_ticks(self, tmp_path, scan):
        fig = _run(tmp_path, scan.assign(Chr=["chr2", "chr1", "1", "chrX", "2", "1"]))
        ax = fig.axes[0]
        assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "2", "X"]

    def test_threshold_lines(self, tmp_path, scan):
        fig = _run(tmp_path, scan, "--sig", "1e-6", "--suggestive", "none")
        heights = [ln.get_ydata()[0] for ln in fig.axes[0].get_lines()]
        assert heights == pytest.approx([6.0])

    def test_ylim_clips_to_triangles(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "--ylim", "0", "20")
        assert fig.axes[0].get_ylim() == (0.0, 20.0)
        below, above = scatters
        assert below[1].max() <= 20
        assert above[1].tolist() == [20.0]
        assert above[2]["marker"] == "^"

    def test_default_ylim_shows_the_line(self, tmp_path):
        df = pd.DataFrame({"Chr": ["1", "1"], "pos": [1, 2], "Pval": [0.5, 0.1]})
        fig = _run(tmp_path, df)
        assert fig.axes[0].get_ylim()[1] > -np.log10(5e-8)

    def test_thinning_keeps_hits(self, tmp_path, scatters, caplog):
        rng = np.random.default_rng(0)
        n = 50_000
        df = pd.DataFrame(
            {
                "Chr": np.repeat(["1", "2"], n // 2),
                "pos": np.tile(np.arange(n // 2), 2),
                "Pval": np.concatenate([rng.uniform(size=n - 3), [1e-9, 1e-12, 1e-30]]),
            }
        )
        with caplog.at_level(logging.INFO):
            _run(tmp_path, df)
        x, y, _ = scatters[0]
        assert len(x) < n
        assert (y >= 2).sum() == (df["Pval"] <= 0.01).sum()
        assert "Thinned" in caplog.text

    def test_groups_get_colours_and_bands(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "-g", "study")
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels == ["a", "b"]
        assert {kw["color"] for _, _, kw in scatters} == {"#E69F00", "#56B4E9"}
        assert len(fig.axes[0].patches) == 1  # band behind chr2 (every other)

    def test_subgraph_panels_share_positions(self, tmp_path, scan, scatters):
        fig = _run(tmp_path, scan, "--subgraphcol", "study", "--no-thin")
        assert [ax.get_title() for ax in fig.axes] == ["a", "b"]
        (xa, _, _), (xb, _, _) = scatters
        # one coordinate system for both panels: chr1 100..300 → 0..200,
        # a gap of 1 % of the 600 bp spanned, chr2 500..900 → 206..606, X 612
        assert xa.tolist() == pytest.approx(
            [206, 0, 606]
        )  # chr2 500, chr1 100, chr2 900
        assert xb.tolist() == pytest.approx([200, 612])  # chr1 300, X 50
        assert fig.axes[0].get_xlim() == fig.axes[1].get_xlim()

    def test_missing_group_column(self, tmp_path, scan):
        with pytest.raises(ValueError, match="-g/--groupcol"):
            _run(tmp_path, scan, "-g", "nosuch")

    def test_example_dataset(self, tmp_path):
        from dftk.common.gwas import simulate_gwas

        fig = _run(tmp_path, simulate_gwas(), "--chi2col", "Chisq")
        assert fig.axes[0].get_ylim()[1] > 80


@pytest.fixture
def two_scans():
    """The same three variants in scans a and b, plus one in c."""
    return pd.DataFrame(
        {
            "Chr": ["1", "1", "2", "1", "1", "2", "1"],
            "pos": [100, 200, 300, 100, 200, 300, 100],
            "Pval": [1e-10, 0.5, 1e-3, 1e-4, 0.1, 1e-30, 0.2],
            "study": ["a", "a", "a", "b", "b", "b", "c"],
        }
    )


class TestMiami:
    def test_two_halves_lower_upside_down(self, tmp_path, two_scans, scatters):
        df = two_scans[two_scans["study"] != "c"]
        fig = _run(tmp_path, df, "--miami", "study", "--no-thin")
        (_, ya, _), (_, yb, _) = scatters
        assert ya.tolist() == pytest.approx([10, -np.log10(0.5), 3])
        assert yb.tolist() == pytest.approx([4, 1, 30])
        upper, lower = fig.axes
        assert upper.get_ylim()[0] == 0 and 10 < upper.get_ylim()[1] < 30
        lo, hi = lower.get_ylim()
        assert hi == 0 and lo > 30  # inverted: 0 at the top
        assert upper.texts[-1].get_text() == "a"
        assert [t.get_text() for t in lower.texts] == ["b"]
        # threshold lines in both halves
        for ax in fig.axes:
            heights = sorted(ln.get_ydata()[0] for ln in ax.get_lines())
            assert heights == pytest.approx([5, -np.log10(5e-8)])

    def test_gap_holds_the_chromosome_numbers(self, tmp_path, two_scans):
        fig = _run(tmp_path, two_scans, "--miami", "study", "a", "b")
        upper, lower = fig.axes
        fig.canvas.draw()
        top, bottom = upper.get_position().y0, lower.get_position().y1
        assert top > bottom  # a gap
        numbers = upper.texts[:-1]
        assert [t.get_text() for t in numbers] == ["1", "2"]
        # centred in the gap (figure coordinates)
        renderer = fig.canvas.get_renderer()
        for t in numbers:
            box = t.get_window_extent(renderer).transformed(fig.transFigure.inverted())
            assert (box.y0 + box.y1) / 2 == pytest.approx((top + bottom) / 2, abs=0.01)
            assert box.y0 > bottom and box.y1 < top
        assert upper.get_xlim() == lower.get_xlim()

    def test_halves_on_one_scale(self, tmp_path, two_scans):
        df = two_scans[two_scans["study"] != "c"]
        fig = _run(tmp_path, df, "--miami", "study")
        upper, lower = fig.axes
        per_unit = [
            ax.get_position().height / abs(np.diff(ax.get_ylim())[0])
            for ax in (upper, lower)
        ]
        assert per_unit[0] == pytest.approx(per_unit[1], rel=0.01)

    def test_named_values_pick_and_order(self, tmp_path, two_scans, scatters):
        fig = _run(tmp_path, two_scans, "--miami", "study", "b", "a", "--no-thin")
        (_, yb, _), (_, ya, _) = scatters
        assert yb.tolist() == pytest.approx([4, 1, 30])  # c left out
        assert fig.axes[0].texts[-1].get_text() == "b"

    def test_ylim_caps_each_half(self, tmp_path, two_scans, scatters):
        df = two_scans[two_scans["study"] != "c"]
        fig = _run(tmp_path, df, "--miami", "study", "--ylim", "-20", "8")
        upper, lower = fig.axes
        assert upper.get_ylim() == (0.0, 8.0) and lower.get_ylim() == (20.0, 0.0)
        _, a_capped, b_inside, b_capped = scatters
        assert a_capped[1].tolist() == [8.0] and a_capped[2]["marker"] == "^"
        assert b_capped[1].tolist() == [20.0] and b_capped[2]["marker"] == "v"

    def test_title_on_the_upper_half(self, tmp_path, two_scans):
        fig = _run(tmp_path, two_scans, "--miami", "study", "a", "b", "-t", "T")
        assert fig.axes[0].get_title() == "T"

    def test_panels(self, tmp_path, two_scans):
        df = pd.concat([two_scans.assign(set=1), two_scans.assign(set=2)])
        fig = _run(tmp_path, df, "--miami", "study", "a", "b", "--subgraphcol", "set")
        assert len(fig.axes) == 4
        assert [ax.get_title() for ax in fig.axes[::2]] == ["1", "2"]

    def test_more_than_two_values_needs_names(self, tmp_path, two_scans):
        with pytest.raises(ValueError, match="holds 3 values"):
            _run(tmp_path, two_scans, "--miami", "study")

    def test_unknown_value(self, tmp_path, two_scans):
        with pytest.raises(ValueError, match="'x'"):
            _run(tmp_path, two_scans, "--miami", "study", "a", "x")

    def test_wrong_number_of_arguments(self, tmp_path, two_scans):
        with pytest.raises(ValueError, match="not 2 arguments"):
            _run(tmp_path, two_scans, "--miami", "study", "a")

    def test_not_with_groupcol(self, tmp_path, two_scans):
        with pytest.raises(ValueError, match="can't be combined"):
            _run(tmp_path, two_scans, "--miami", "study", "a", "b", "-g", "study")

    def test_taller_default_size(self, tmp_path, two_scans):
        fig = _run(tmp_path, two_scans, "--miami", "study", "a", "b")
        assert tuple(fig.get_size_inches()) == pytest.approx((12, 6))


class TestLogy:
    def test_symlog_axis_and_ticks(self, tmp_path, scan):
        fig = _run(tmp_path, scan, "--logy")
        ax = fig.axes[0]
        assert ax.get_yscale() == "symlog"
        assert ax.yaxis.get_transform().linthresh == 10
        # linear part, then 1-2-5 steps up to the 1e-40 peak
        assert ax.get_yticks().tolist() == [0, 5, 10, 20]
        assert ax.yaxis.get_major_formatter()(20, 0) == "20"

    def test_threshold_value(self, tmp_path, scan):
        fig = _run(tmp_path, scan, "--logy", "3")
        assert fig.axes[0].yaxis.get_transform().linthresh == 3

    def test_only_powers_of_ten_on_a_tall_axis(self, tmp_path, scan):
        df = scan.assign(m=[3000, 1, 2, 0.5, 400, 1])
        fig = _run(tmp_path, df, "--mlogpcol", "m", "--logy")
        log = [t for t in fig.axes[0].get_yticks() if t > 10]
        assert log == [100, 1000]

    def test_with_miami(self, tmp_path, two_scans):
        fig = _run(tmp_path, two_scans, "--miami", "study", "a", "b", "--logy")
        upper, lower = fig.axes
        assert upper.get_yticks().tolist() == [0, 5, 10]
        assert lower.get_yticks().tolist() == [0, 5, 10, 20]
        assert lower.get_ylim()[1] == 0

    def test_bad_threshold(self, tmp_path, scan):
        with pytest.raises(ValueError, match="--logy"):
            _run(tmp_path, scan, "--logy", "0")


class TestSigcolor:
    def test_points_past_the_line(self, tmp_path, scan, scatters):
        _run(tmp_path, scan, "--no-thin", "--sigcolor", "red")
        _, y, kw = scatters[0]
        red = np.array([c == "red" for c in kw["c"]])
        assert red.tolist() == (y >= -np.log10(5e-8)).tolist()
