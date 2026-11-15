"""
Tests for dftk qq.  Figures are saved with the Agg backend; the points drawn
are checked through the scatter calls.
"""

import argparse
import logging
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.axes  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import scipy.stats as ss  # noqa: E402

import dftk.commands.qq_cmd as qq_mod  # noqa: E402
from dftk.commands.qq_cmd import QqCommand, qq_table  # noqa: E402


def _args(*argv):
    parser = argparse.ArgumentParser()
    QqCommand().add_arguments(parser)
    return parser.parse_args(list(argv))


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
    data = tmp_path / "in.tsv"
    df.to_csv(data, sep="\t", index=False)
    png = tmp_path / "out.png"
    saved = []
    orig = qq_mod.save_or_show
    with patch.object(
        qq_mod, "save_or_show", lambda f, a: saved.append(f) or orig(f, a)
    ):
        QqCommand().execute(_args(str(data), *argv, "-f", str(png)))
    assert png.stat().st_size > 0
    return saved[0]


def _texts(fig):
    return [t.get_text() for t in fig.axes[0].texts]


@pytest.fixture
def null():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"Pval": rng.uniform(size=5000)})


class TestQqTable:
    def test_missing_p_left_out(self):
        df = pd.DataFrame({"Pval": [0.5, np.nan, 0.01]})
        assert qq_table(df, _args())["_y"].tolist() == pytest.approx([np.log10(2), 2])

    def test_no_position_options(self):
        with pytest.raises(SystemExit):
            _args("--chrcol", "Chr")

    def test_nothing_left_is_an_error(self):
        with pytest.raises(ValueError, match="No rows"):
            qq_table(pd.DataFrame({"Pval": [np.nan]}), _args())


class TestExecute:
    def test_points_sorted_against_expected(self, tmp_path, scatters):
        df = pd.DataFrame({"Pval": [0.5, 0.001, 0.2]})
        _run(tmp_path, df, "--no-thin")
        x, y, _ = scatters[0]
        assert x == pytest.approx(-np.log10([0.25, 0.5, 0.75]))
        assert y == pytest.approx(-np.log10([0.001, 0.2, 0.5]))

    def test_null_lambda_on_plot(self, tmp_path, null):
        fig = _run(tmp_path, null)
        (text,) = _texts(fig)
        assert text.startswith("λ = ")
        assert float(text.split("= ")[1]) == pytest.approx(1.0, abs=0.1)

    def test_no_lambda(self, tmp_path, null, caplog):
        with caplog.at_level(logging.INFO):
            fig = _run(tmp_path, null, "--no-lambda")
        assert _texts(fig) == []
        assert "lambda = " in caplog.text

    def test_chi2_inflated(self, tmp_path):
        chi2 = ss.chi2.rvs(1, size=20_000, random_state=3) * 1.3
        fig = _run(tmp_path, pd.DataFrame({"stat": chi2}), "--chi2col", "stat")
        assert float(_texts(fig)[0].split("= ")[1]) == pytest.approx(1.3, abs=0.05)

    def test_same_from_p_mlogp_and_chi2(self, tmp_path, scatters):
        chi2 = np.array([0.1, 2.0, 9.0, 40.0])
        p = ss.chi2.sf(chi2, 1)
        df = pd.DataFrame({"Pval": p, "m": -np.log10(p), "c": chi2})
        for opts in ([], ["--mlogpcol", "m"], ["--chi2col", "c"]):
            _run(tmp_path, df, "--no-thin", *opts)
        ys = [y for _, y, _ in scatters]
        assert ys[0] == pytest.approx(ys[1]) and ys[0] == pytest.approx(ys[2])

    def test_band_and_diagonal(self, tmp_path, null):
        fig = _run(tmp_path, null)
        ax = fig.axes[0]
        assert len(ax.collections) == 2  # band + points
        (diag,) = ax.get_lines()
        assert list(diag.get_xdata()) == list(diag.get_ydata())

    def test_no_band(self, tmp_path, null):
        fig = _run(tmp_path, null, "--no-ci")
        assert len(fig.axes[0].collections) == 1

    def test_bad_ci(self, tmp_path, null):
        with pytest.raises(ValueError, match="--ci"):
            _run(tmp_path, null, "--ci", "95")

    def test_thinning_keeps_the_tail(self, tmp_path, scatters):
        p = np.random.default_rng(1).uniform(size=100_000)
        _run(tmp_path, pd.DataFrame({"Pval": p}))
        _, y, _ = scatters[0]
        assert len(y) < len(p)
        assert (y >= 2).sum() == (p <= 0.01).sum()

    def test_ylim_triangles(self, tmp_path, scatters):
        df = pd.DataFrame({"Pval": [1e-30, 1e-3, 0.5]})
        fig = _run(tmp_path, df, "--ylim", "0", "10", "--no-thin")
        assert fig.axes[0].get_ylim() == (0.0, 10.0)
        _, above_y, kw = scatters[1]
        assert above_y.tolist() == [10.0] and kw["marker"] == "^"

    def test_groups_each_own_expected_and_lambda(self, tmp_path, scatters):
        df = pd.DataFrame(
            {"Pval": [0.5, 0.1, 0.01, 0.9], "study": ["a", "a", "a", "b"]}
        )
        fig = _run(tmp_path, df, "-g", "study", "--no-thin")
        (xa, _, _), (xb, _, _) = scatters
        assert len(xa) == 3 and xb == pytest.approx([-np.log10(0.5)])
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        assert labels[0].startswith("a (λ = ") and labels[1].startswith("b (λ = ")

    def test_subgraph_panels(self, tmp_path):
        df = pd.DataFrame({"Pval": [0.5, 0.1, 0.01, 0.9], "study": list("aabb")})
        fig = _run(tmp_path, df, "--subgraphcol", "study")
        assert [ax.get_title() for ax in fig.axes] == ["a", "b"]

    def test_example_dataset(self, tmp_path):
        from dftk.common.gwas import simulate_gwas

        fig = _run(tmp_path, simulate_gwas(), "--chi2col", "Chisq")
        assert float(_texts(fig)[0].split("= ")[1]) < 1.05


class TestLogy:
    def test_symlog_axis(self, tmp_path, null):
        df = pd.concat([null, pd.DataFrame({"Pval": [1e-60]})])
        fig = _run(tmp_path, df, "--logy")
        ax = fig.axes[0]
        assert ax.get_yscale() == "symlog"
        assert 50 in ax.get_yticks()
        assert ax.yaxis.get_major_formatter()(50, 0) == "50"

    def test_bad_threshold(self, tmp_path, null):
        with pytest.raises(ValueError, match="--logy"):
            _run(tmp_path, null, "--logy", "-1")
