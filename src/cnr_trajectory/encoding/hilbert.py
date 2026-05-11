"""Hilbert curve encoding utilities.

This module preserves the recursive Hilbert-index implementation from the
original notebook helper `Hilbert_function.py`.
"""


def hilbert_index(x: float, y: float, eps: float) -> float:
    """Return the Hilbert index for a point in the unit square.

    Parameters
    ----------
    x, y:
        Coordinates of an image point. The original experiment assumes both
        values are in ``[0, 1]``.
    eps:
        Required precision of the index parameter.
    """
    if eps > 1:
        return 0

    if x < 0.5:
        if y < 0.5:
            return (0 + hilbert_index(2 * y, 2 * x, 4 * eps)) / 4
        return (1 + hilbert_index(2 * x, 2 * y - 1, 4 * eps)) / 4

    if y >= 0.5:
        return (2 + hilbert_index(2 * x - 1, 2 * y - 1, 4 * eps)) / 4
    return (3 + hilbert_index(1 - 2 * y, 2 - 2 * x, 4 * eps)) / 4


def hilbIndex(x: float, y: float, eps: float) -> float:
    """Backward-compatible alias for the original notebook function name."""
    return hilbert_index(x, y, eps)
