"""
Tests for dftk phewas.  Figures are saved with the Agg backend; what was
drawn is read back from the axes.
"""

import argparse
import logging
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import dftk.commands.phewas_cmd as phewas_mod  # noqa: E402
from dftk.commands.phewas_cmd import PhewasCommand, layout  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    PhewasCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


def _run(tmp_path, df, *argv):
    """Run the command to a PNG; return the axes it drew."""
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = phewas_mod.save_or_show
    with patch.object(
        phewas_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        PhewasCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0].axes[0]


def _points(ax):
    """[(marker path vertex count, x, y)] of every point drawn."""
    out = []
    for c in ax.collections:
        for x, y in c.get_offsets():
            out.append((len(c.get_paths()[0].vertices), float(x), float(y)))
    return out


@pytest.fixture
def results():
    return pd.DataFrame(
        {
            "category": ["Heart", "Heart", "Blood", "Heart", "Eye"],
            "phenotype": ["AF", "CAD", "Anaemia", "Stroke", "Cataract"],
            "Pval": [1e-8, 1e-6, 0.5, 0.2, 1e-20],
            "beta": [0.3, -0.2, 0.0, 0.1, 0.4],
        }
    )


class TestLayout:
    def test_categories_in_order_of_first_appearance(self, results):
        data = results.assign(_cat=results["category"])
        x, spans = layout(data)
        assert [c for c, _, _ in spans] == ["Heart", "Blood", "Eye"]
        # Heart rows keep their file order
        assert x[0] < x[1] < x[3] < x[2] < x[4]

    def test_equal_widths(self, results):
        """Every category as wide, its phenotypes spread evenly across it."""
        data = results.assign(_cat=results["category"])
        x, spans = layout(data)
        widths = {round(b - a, 9) for _, a, b in spans}
        assert widths == {1.0}
        heart = spans[0]
        assert list(x[[0, 1, 3]]) == pytest.approx([1 / 6, 3 / 6, 5 / 6])
        # One phenotype stands in the middle of its band
        for i, (_, a, b) in ((2, spans[1]), (4, spans[2])):
            assert x[i] == pytest.approx((a + b) / 2)
        assert heart[2] < spans[1][1]  # a gap between bands

    def test_count_widths(self, results):
        """--catwidth count: one unit per phenotype, as before 0.27.0."""
        data = results.assign(_cat=results["category"])
        x, spans = layout(data, equal=False)
        assert list(x) == [0.5, 1.5, 4.5, 2.5, 6.5]
        assert [(a, b) for _, a, b in spans] == [(0, 3), (4, 5), (6, 7)]

    def test_catwidth_option(self, tmp_path, results):
        def xs(*argv):
            return sorted(x for _, x, _ in _points(_run(tmp_path, results, *argv)))

        assert xs() == pytest.approx([1 / 6, 3 / 6, 5 / 6, 1.65, 2.8])
        assert xs("--catwidth", "count") == [0.5, 1.5, 2.5, 4.5, 6.5]

    def test_tick_labels_and_y(self, tmp_path, results):
        ax = _run(tmp_path, results)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["Heart", "Blood", "Eye"]
        ys = sorted(y for _, _, y in _points(ax))
        assert ys == pytest.approx(sorted(-np.log10(results["Pval"])))

    def test_category_colours_differ(self, tmp_path, results):
        ax = _run(tmp_path, results)
        colours = {t.get_text(): t.get_color() for t in ax.get_xticklabels()}
        assert len(set(colours.values())) == 3


class TestDirection:
    def test_triangles_by_sign(self, tmp_path, results):
        from matplotlib.markers import MarkerStyle

        def shape(collection):
            verts = collection.get_paths()[0].vertices
            for m in ("^", "v", "o"):
                ms = MarkerStyle(m)
                ref = ms.get_path().transformed(ms.get_transform()).vertices
                if ref.shape == verts.shape and np.allclose(ref, verts):
                    return m
            return None

        ax = _run(tmp_path, results, "--betacol", "beta")
        marker_at = {
            round(float(y), 3): shape(c)
            for c in ax.collections
            for _, y in c.get_offsets()
        }
        assert marker_at[8.0] == "^"  # AF, beta > 0
        assert marker_at[6.0] == "v"  # CAD, beta < 0
        assert marker_at[round(-np.log10(0.5), 3)] == "o"  # Anaemia, beta 0
        texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert texts == ["beta > 0", "beta < 0"]

    def test_no_betacol_no_legend(self, tmp_path, results):
        assert _run(tmp_path, results).get_legend() is None

    def test_non_numeric_beta(self, tmp_path, results):
        with pytest.raises(ValueError, match="isn't numeric"):
            _run(tmp_path, results, "--betacol", "phenotype")


class TestLines:
    def test_bonferroni_default(self, tmp_path, results):
        ax = _run(tmp_path, results)
        (line,) = ax.lines
        assert line.get_ydata()[0] == pytest.approx(-np.log10(0.05 / 5))

    def test_fixed_and_suggestive(self, tmp_path, results):
        ax = _run(tmp_path, results, "--sig", "1e-5", "--suggestive", "1e-3")
        assert sorted(ln.get_ydata()[0] for ln in ax.lines) == pytest.approx([3, 5])

    def test_none(self, tmp_path, results):
        ax = _run(tmp_path, results, "--sig", "none")
        assert len(ax.lines) == 0
        assert len(ax.texts) == 0  # no line, so nothing counts as a hit

    def test_bad_sig(self):
        with pytest.raises(SystemExit):
            _args("--sig", "2")


class TestLabels:
    def test_hits_above_line_strongest_first(self, tmp_path, results):
        ax = _run(tmp_path, results)
        assert [t.get_text() for t in ax.texts] == ["Cataract", "AF", "CAD"]

    def test_label_limit(self, tmp_path, results, caplog):
        with caplog.at_level(logging.INFO):
            ax = _run(tmp_path, results, "--label", "1")
        assert [t.get_text() for t in ax.texts] == ["Cataract"]
        assert "Labelled 1 of 3 phenotypes above the line (--label 1)" in caplog.text
        assert len(_run(tmp_path, results, "--label", "0").texts) == 0

    def test_labels_do_not_overlap(self, tmp_path):
        df = pd.DataFrame(
            {
                "category": ["A"] * 4,
                "phenotype": ["first long name", "second long name", "x", "y"],
                "Pval": [1e-9, 2e-9, 3e-9, 0.5],
            }
        )
        ax = _run(tmp_path, df)
        renderer = ax.figure.canvas.get_renderer()
        boxes = [t.get_window_extent(renderer) for t in ax.texts]
        assert len(boxes) == 3
        for i in range(len(boxes)):
            for j in range(i):
                assert not boxes[i].overlaps(boxes[j])

    def test_label_per_category(self, tmp_path, results, caplog):
        """AF and CAD are both Heart hits: only the stronger is labelled."""
        with caplog.at_level(logging.INFO):
            ax = _run(tmp_path, results, "--labelpercat", "1")
        assert [t.get_text() for t in ax.texts] == ["Cataract", "AF"]
        assert "Labelled 2 of 3" in caplog.text and "--labelpercat 1" in caplog.text
        texts = _run(tmp_path, results, "--labelpercat", "2").texts
        assert [t.get_text() for t in texts] == ["Cataract", "AF", "CAD"]

    def test_label_per_category_and_total(self, tmp_path, results):
        ax = _run(tmp_path, results, "--labelpercat", "1", "--label", "1")
        assert [t.get_text() for t in ax.texts] == ["Cataract"]

    def test_bad_label_per_category(self, tmp_path, results):
        with pytest.raises(ValueError, match="--labelpercat must be 1 or more"):
            _run(tmp_path, results, "--labelpercat", "0")


class TestNumbers:
    """--labelnumbers: numbers left to right, the key on stdout."""

    @staticmethod
    def _key(capsys):
        import io

        return pd.read_csv(io.StringIO(capsys.readouterr().out), sep="\t")

    def test_numbered_left_to_right(self, tmp_path, results, capsys):
        ax = _run(tmp_path, results, "--labelnumbers", "--betacol", "beta")
        # Labelled strongest first; numbered by x: AF, CAD (Heart), Cataract (Eye)
        number = {t.get_text(): t.xy[0] for t in ax.texts}
        assert sorted(number, key=number.get) == ["1", "2", "3"]
        key = self._key(capsys)
        assert list(key.columns) == ["number", "category", "phenotype", "Pval", "beta"]
        assert list(key["phenotype"]) == ["AF", "CAD", "Cataract"]
        assert list(key["Pval"]) == pytest.approx([1e-8, 1e-6, 1e-20])
        assert list(key["beta"]) == [0.3, -0.2, 0.4]

    def test_with_label_per_category(self, tmp_path, results, capsys):
        ax = _run(tmp_path, results, "--labelnumbers", "--labelpercat", "1")
        assert sorted(t.get_text() for t in ax.texts) == ["1", "2"]
        key = self._key(capsys)
        assert list(zip(key["number"], key["phenotype"], strict=True)) == [
            (1, "AF"),
            (2, "Cataract"),
        ]

    def test_named_columns_in_key(self, tmp_path, results, capsys):
        df = results.rename(columns={"category": "group", "phenotype": "trait"})
        _run(tmp_path, df, "--labelnumbers", "-c", "group", "-l", "trait")
        assert list(self._key(capsys).columns) == ["number", "group", "trait", "Pval"]

    def test_nothing_above_the_line(self, tmp_path, results, capsys, caplog):
        with caplog.at_level(logging.INFO):
            ax = _run(tmp_path, results, "--labelnumbers", "--sig", "1e-30")
        assert len(ax.texts) == 0
        assert capsys.readouterr().out == ""
        assert "no phenotype above the line" in caplog.text

    def test_without_option_nothing_on_stdout(self, tmp_path, results, capsys):
        _run(tmp_path, results)
        assert capsys.readouterr().out == ""


class TestColumns:
    def test_named_columns_and_mlogp(self, tmp_path, results):
        df = results.rename(columns={"category": "grp", "phenotype": "name"})
        df["mlogp"] = -np.log10(df.pop("Pval"))
        ax = _run(tmp_path, df, "-c", "grp", "-l", "name", "--mlogpcol", "mlogp")
        assert {t.get_text() for t in ax.texts} == {"Cataract", "AF", "CAD"}

    def test_p_value_alias(self, tmp_path, results):
        ax = _run(tmp_path, results.rename(columns={"Pval": "P"}))
        assert len(_points(ax)) == 5

    def test_missing_category_column(self, tmp_path, results):
        with pytest.raises(ValueError, match="nosuch"):
            _run(tmp_path, results, "-c", "nosuch")

    def test_rows_without_p_left_out(self, tmp_path, results):
        results.loc[2, "Pval"] = np.nan
        assert len(_points(_run(tmp_path, results))) == 4

    def test_logy(self, tmp_path, results):
        ax = _run(tmp_path, results, "--logy", "5")
        assert ax.get_yscale() == "symlog"

    def test_ylim_caps(self, tmp_path, results):
        ax = _run(tmp_path, results, "-ym", "0", "10")
        assert max(y for _, _, y in _points(ax)) == pytest.approx(10)


class TestSubgraph:
    """--subgraphcol: one panel per value, shared x positions and y axis."""

    @pytest.fixture
    def models(self, results):
        # The lof model leaves out Anaemia (untested) and has a weaker AF
        lof = results.drop(index=2).assign(model="lof")
        lof.loc[0, "Pval"] = 0.05
        return pd.concat([results.assign(model="all"), lof], ignore_index=True)

    def _fig(self, tmp_path, df, *argv):
        data = tmp_path / "in.tsv"
        df.to_csv(data, sep="\t", index=False)
        png = tmp_path / "out.png"
        saved = []
        orig = phewas_mod.save_or_show
        with patch.object(
            phewas_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
        ):
            PhewasCommand().execute(_args(str(data), *argv, "-f", str(png)))
        return saved[0]

    @staticmethod
    def _panels(fig):
        return [ax for ax in fig.axes if ax.get_visible()]

    @staticmethod
    def _xy(ax):
        return {(x, round(y, 3)) for _, x, y in _points(ax)}

    def test_stacked_with_titles(self, tmp_path, models):
        fig = self._fig(tmp_path, models, "--subgraphcol", "model")
        top, bottom = self._panels(fig)
        assert (top.get_title(), bottom.get_title()) == ("all", "lof")
        assert top.get_position().y0 > bottom.get_position().y1  # one per row

    def test_same_x_for_a_phenotype_in_every_panel(self, tmp_path, models):
        fig = self._fig(tmp_path, models, "--subgraphcol", "model")
        top, bottom = self._panels(fig)
        xs_top = {round(y, 3): x for _, x, y in _points(top)}
        xs_bottom = {round(y, 3): x for _, x, y in _points(bottom)}
        # Cataract (p 1e-20) and Stroke (p 0.2) at the same x in both panels
        assert xs_top[20.0] == xs_bottom[20.0]
        assert xs_top[round(-np.log10(0.2), 3)] == xs_bottom[round(-np.log10(0.2), 3)]
        # Anaemia untested in lof: a gap, not a shift
        assert len(_points(bottom)) == len(_points(top)) - 1
        assert top.get_xlim() == bottom.get_xlim()

    def test_one_y_axis(self, tmp_path, models):
        fig = self._fig(tmp_path, models, "--subgraphcol", "model")
        top, bottom = self._panels(fig)
        assert top.get_ylim() == bottom.get_ylim()

    def test_bonferroni_per_panel(self, tmp_path, models, caplog):
        with caplog.at_level(logging.INFO, logger=phewas_mod.logger.name):
            fig = self._fig(tmp_path, models, "--subgraphcol", "model")
        top, bottom = self._panels(fig)
        line = lambda ax: [ln.get_ydata()[0] for ln in ax.lines]  # noqa: E731
        assert line(top) == [pytest.approx(-np.log10(0.05 / 5))]
        assert line(bottom) == [pytest.approx(-np.log10(0.05 / 4))]
        assert "all: 0.05/5 = 0.01; lof: 0.05/4 = 0.013" in caplog.text

    def test_fixed_sig_same_in_every_panel(self, tmp_path, models, caplog):
        with caplog.at_level(logging.INFO, logger=phewas_mod.logger.name):
            fig = self._fig(tmp_path, models, "--subgraphcol", "model", "--sig", "1e-5")
        heights = [ln.get_ydata()[0] for ax in self._panels(fig) for ln in ax.lines]
        assert heights == [pytest.approx(5.0)] * 2
        assert "Bonferroni line per panel" not in caplog.text

    def test_labels_per_panel(self, tmp_path, models):
        fig = self._fig(tmp_path, models, "--subgraphcol", "model")
        top, bottom = self._panels(fig)
        assert [t.get_text() for t in top.texts] == ["Cataract", "AF", "CAD"]
        # AF is p 0.05 in lof, below that panel's line
        assert [t.get_text() for t in bottom.texts] == ["Cataract", "CAD"]

    def test_numbers_shared_across_panels(self, tmp_path, models, capsys):
        """A phenotype labelled in both panels keeps one number."""
        fig = self._fig(tmp_path, models, "--subgraphcol", "model", "--labelnumbers")
        all_, lof = self._panels(fig)
        assert sorted(t.get_text() for t in all_.texts) == ["1", "2", "3"]
        assert sorted(t.get_text() for t in lof.texts) == ["2", "3"]
        import io

        key = pd.read_csv(io.StringIO(capsys.readouterr().out), sep="\t")
        assert list(key.columns) == ["number", "model", "category", "phenotype", "Pval"]
        assert list(
            zip(key["number"], key["model"], key["phenotype"], strict=True)
        ) == [
            (1, "all", "AF"),
            (2, "all", "CAD"),
            (2, "lof", "CAD"),
            (3, "all", "Cataract"),
            (3, "lof", "Cataract"),
        ]

    def test_label_per_category_per_panel(self, tmp_path, models, caplog):
        with caplog.at_level(logging.INFO):
            fig = self._fig(
                tmp_path, models, "--subgraphcol", "model", "--labelpercat", "1"
            )
        all_, lof = self._panels(fig)
        assert [t.get_text() for t in all_.texts] == ["Cataract", "AF"]
        assert [t.get_text() for t in lof.texts] == ["Cataract", "CAD"]
        assert "all: Labelled 2 of 3" in caplog.text

    def test_legend_and_categories_once(self, tmp_path, models):
        fig = self._fig(tmp_path, models, "--subgraphcol", "model", "--betacol", "beta")
        top, bottom = self._panels(fig)
        assert top.get_legend() is not None and bottom.get_legend() is None
        assert all(t.get_text() == "" for t in top.get_xticklabels())
        assert [t.get_text() for t in bottom.get_xticklabels()] == [
            "Heart",
            "Blood",
            "Eye",
        ]

    def test_grid_order_and_format(self, tmp_path, models):
        df = pd.concat([models, models[models.model == "all"].assign(model="x")])
        fig = self._fig(
            tmp_path, df, "--subgraphcol", "model", "--ncols", "2",
            "--subgraphorder", "lof", "--subgraphformat", "Model: {model}",
        )  # fmt: skip
        panels = self._panels(fig)
        assert [ax.get_title() for ax in panels] == [
            "Model: lof",
            "Model: all",
            "Model: x",
        ]
        assert sum(not ax.get_visible() for ax in fig.axes) == 1
        # the right-hand panel (above the empty cell) carries the categories
        assert [t.get_text() for t in panels[1].get_xticklabels()][0] == "Heart"
        assert panels[1].get_ylabel() == ""

    @pytest.mark.parametrize("col", ["category", "phenotype"])
    def test_subgraphcol_cannot_be_category_or_label(self, tmp_path, models, col):
        with pytest.raises(ValueError, match="is also -c or -l"):
            self._fig(tmp_path, models, "--subgraphcol", col)

    def test_missing_subgraphcol(self, tmp_path, models):
        with pytest.raises(ValueError, match="nosuch"):
            self._fig(tmp_path, models, "--subgraphcol", "nosuch")
