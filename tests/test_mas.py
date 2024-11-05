# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

"""Tests for monotonic alignment search."""

import torch

from monotonic_alignment_search import maximum_path


def test_mas() -> None:
    """Basic functionality test."""
    for shape in [(1, 20, 40), (10, 20, 40)]:
        value = torch.rand(shape)
        mask = torch.ones_like(value)
        maximum_path(value, mask)
