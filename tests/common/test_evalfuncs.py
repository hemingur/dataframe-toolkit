"""Tests for dftk.common.evalfuncs: deciding which formulas are special calls."""

import pytest

from dftk.common.evalfuncs import is_special_call


@pytest.mark.parametrize(
    "formula",
    [
        "z = sum(a, b)",
        "z=mean(a,b)",
        "  z = where(big, 'y', 'n')  ",
        "z = jn(2, x)",
        "z = colsum(a*)",
        "z = binom_test(k, n, p)",
    ],
)
def test_special_calls(formula):
    assert is_special_call(formula)


@pytest.mark.parametrize(
    "formula",
    [
        "z = a + b",
        "z = sum(a, b) * 2",  # the call must be the whole right-hand side
        "z = sum(a, b) + max(c, d)",
        "z = log1p(x)",  # built into pandas df.eval
        "z = expm1(x)",
        "z = frobnicate(a)",  # not a dftk function
        "z = a.fillna(b)",
        "z = sum(a, b",  # unbalanced: pandas reports the syntax error
        "sum(a, b)",  # no assignment
    ],
)
def test_not_special_calls(formula):
    assert not is_special_call(formula)
