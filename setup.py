# SPDX-FileCopyrightText: Enno Hermann
#
# SPDX-License-Identifier: MIT

"""Builds the Cython extension."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

exts = [
    Extension(
        name="monotonic_alignment_search.core",
        sources=["src/monotonic_alignment_search/core.pyx"],
    ),
]
setup(
    include_dirs=np.get_include(),
    ext_modules=cythonize(exts, language_level=3),
    zip_safe=False,
)
