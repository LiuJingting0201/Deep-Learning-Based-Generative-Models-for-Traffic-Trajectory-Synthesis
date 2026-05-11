"""Sequence length utilities used before trajectory image encoding.

The functions in this module preserve the behavior of the original helper
files:

- `US_function.py`
- `Squeeze_function.py`
- `Duplicate_function.py`
"""

from __future__ import annotations

import math
from typing import Iterable


def uniform_sampling(values: list[float], final_length: int) -> list[float]:
    """Sample a list uniformly when it is longer than the target length."""
    if len(values) > final_length:
        step = math.floor(len(values) / final_length)
        return values[0 : step * final_length : step]
    return values


def ceil_list(values: list[float], final_length: int) -> list[float]:
    """Preserve original `ceil_list` compression behavior."""
    if len(values) <= final_length:
        return values

    old_length = len(values)
    final_list: list[float] = []
    seg_length = 1
    for index, value in enumerate(values):
        if index != len(values) - 1:
            if value == values[index + 1]:
                seg_length += 1
            else:
                final_list = final_list + [value] * math.ceil(
                    seg_length * final_length / old_length
                )
                seg_length = 1
        else:
            final_list = final_list + [value] * math.ceil(
                seg_length * final_length / old_length
            )
            seg_length = 1
    return final_list


def index_and_length(values: list[float]) -> tuple[list[int], list[int], list[float]]:
    """Return run start indices and lengths for consecutive equal values."""
    list_start_index: list[int] = []
    list_length: list[int] = []
    seg_length = 1
    for index, value in enumerate(values):
        if index != len(values) - 1:
            if value == values[index + 1]:
                seg_length += 1
            else:
                list_start_index.append(index + 1 - seg_length)
                list_length.append(seg_length)
                seg_length = 1
        else:
            list_start_index.append(index + 1 - seg_length)
            list_length.append(seg_length)
            seg_length = 1
    return list_start_index, list_length, values


def squeeze(values: list[float], final_length: int) -> list[float]:
    """Compress a list to a target length using the original notebook method."""
    values = ceil_list(values, final_length)
    list_start_index, list_length, values = index_and_length(values)

    index = sorted(range(len(list_length)), key=lambda k: list_length[k], reverse=True)
    list_start_index = [list_start_index[i] for i in index]
    list_length = [list_length[i] for i in index]

    list_length_updated = list_length.copy()
    j = 0
    while (_nan_count(values) < len(values) - final_length) and (len(values) > final_length):
        if all(item == 1 for item in list_length_updated) is True:
            print("The list cannot be compressed")
            break
        if list_length_updated[j] > 1:
            removed_index = list_start_index[j]
            values[removed_index] = float("nan")
            list_start_index.append(list_start_index[j] + 1)
            list_length_updated.append(list_length_updated[j] - 1)
            list_start_index.pop(0)
            list_length_updated.pop(0)
        else:
            list_start_index.append(list_start_index[j])
            list_length_updated.append(list_length_updated[j])
            list_start_index.pop(0)
            list_length_updated.pop(0)
    return list(filter(lambda x: not math.isnan(x), values))


def duplicate(values: list[float], k: int) -> list[float]:
    """Mirror-duplicate a list until it reaches the target length."""
    list_new = values
    if len(list_new) > k:
        print("The list exceeds the limit")
        return list_new
    while len(list_new) < k:
        list_new = list_new + list_new[::-1]
    return list_new[0:k]


def duplicate_original_name(values: list[float], K: int) -> list[float]:
    """Backward-compatible wrapper for the original capitalized argument name."""
    return duplicate(values, K)


def ensure_list(values: Iterable[float]) -> list[float]:
    """Convert an iterable to a list for downstream notebook-compatible helpers."""
    return list(values)


def _nan_count(values: list[float]) -> int:
    """Count NaN values without requiring NumPy at import time."""
    return sum(1 for value in values if isinstance(value, float) and math.isnan(value))
