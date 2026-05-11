"""Behavior checks for extracted notebook helper functions."""

from cnr_trajectory.encoding.hilbert import hilbIndex, hilbert_index
from cnr_trajectory.encoding.sequence import duplicate, squeeze, uniform_sampling


def test_hilbert_index_keeps_original_alias() -> None:
    assert hilbIndex(0.25, 0.25, 2.0) == hilbert_index(0.25, 0.25, 2.0)


def test_duplicate_mirrors_until_target_length() -> None:
    assert duplicate([1, 2], 5) == [1, 2, 2, 1, 1]


def test_uniform_sampling_short_list_is_unchanged() -> None:
    values = [1, 2, 3]
    assert uniform_sampling(values, 5) == values


def test_uniform_sampling_long_list_uses_original_step_logic() -> None:
    assert uniform_sampling(list(range(10)), 3) == [0, 3, 6]


def test_squeeze_reduces_repeated_segments() -> None:
    assert squeeze([1, 1, 1, 2, 2, 3], 4) == [1, 2, 2, 3]
