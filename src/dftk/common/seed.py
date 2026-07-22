"""
dftk.common.seed — random seed normalisation.

The dftk scripts accept seeds as either integers or arbitrary strings.
Strings are deterministically mapped to 32-bit unsigned integers via MD5
so that friendly names like ``"experiment-1"`` work as reproducible seeds.
"""

import hashlib


def normalize_seed(seed: str | int | None) -> int | None:
    """Return a 32-bit integer seed suitable for ``numpy.random.seed()``.

    * ``None``              → ``None``  (numpy draws from system entropy)
    * integer or digit str  → the integer value
    * any other string      → MD5-hashed to a 32-bit unsigned integer
    """
    if seed is None:
        return None
    try:
        return int(seed)
    except (ValueError, TypeError):
        encoded = str(seed).encode("utf-8")
        return int(hashlib.md5(encoded).hexdigest(), 16) & ((1 << 32) - 1)
