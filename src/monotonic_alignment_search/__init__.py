# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

"""Monotonic alignment search."""

from monotonic_alignment_search.path import (
    maximum_path,
    maximum_path_cython,
    maximum_path_numpy,
)

__all__ = ["maximum_path", "maximum_path_cython", "maximum_path_numpy"]
