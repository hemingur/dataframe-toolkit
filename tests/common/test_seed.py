"""
Tests for stattools.common.seed.normalize_seed.
"""

from stattools.common.seed import normalize_seed


class TestNormalizeSeed:
    def test_none_returns_none(self):
        assert normalize_seed(None) is None

    def test_integer_passthrough(self):
        assert normalize_seed(42) == 42

    def test_zero(self):
        assert normalize_seed(0) == 0

    def test_integer_string(self):
        assert normalize_seed("42") == 42

    def test_negative_integer(self):
        assert normalize_seed(-1) == -1

    def test_string_returns_int(self):
        result = normalize_seed("experiment-1")
        assert isinstance(result, int)

    def test_string_in_32bit_range(self):
        result = normalize_seed("anything")
        assert 0 <= result < 2**32

    def test_string_reproducible(self):
        assert normalize_seed("abc") == normalize_seed("abc")

    def test_different_strings_give_different_seeds(self):
        # MD5 collisions are astronomically unlikely for short strings
        assert normalize_seed("alpha") != normalize_seed("beta")

    def test_empty_string(self):
        result = normalize_seed("")
        assert isinstance(result, int)
        assert 0 <= result < 2**32

    def test_string_differs_from_int_with_same_digits(self):
        # "99" parses to int 99, not hashed
        assert normalize_seed("99") == 99
