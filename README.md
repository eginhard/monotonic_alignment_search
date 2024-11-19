<!--
SPDX-FileCopyrightText: Enno Hermann

SPDX-License-Identifier: MIT
-->

# Monotonic Alignment Search (MAS)

[![PyPI - License](https://img.shields.io/pypi/l/monotonic-alignment-search)](https://github.com/eginhard/monotonic_alignment_search/blob/main/LICENSE)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/monotonic-alignment-search)
[![PyPI - Version](https://img.shields.io/pypi/v/monotonic-alignment-search)](https://pypi.org/project/monotonic-alignment-search)
![GithubActions](https://github.com/eginhard/monotonic_alignment_search/actions/workflows/test.yml/badge.svg)
![GithubActions](https://github.com/eginhard/monotonic_alignment_search/actions/workflows/lint.yml/badge.svg)

Implementation of MAS from [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
for easy reuse in other projects.

## Installation

```bash
pip install monotonic-alignment-search
```

Wheels are provided for Linux, Mac, and Windows. Pytorch is not installed by
default. You either first need to install it yourself, or install one of the
following extras with [uv](https://docs.astral.sh/uv/):

```bash
uv add monotonic-alignment-search[cpu]
uv add monotonic-alignment-search[cuda]
```

## Usage

MAS can find the most probable alignment between a text sequence `t_x` and a
speech sequence `t_y`.

```python
from monotonic_alignment_search import maximum_path

# value (torch.Tensor): [batch_size, t_x, t_y]
# mask  (torch.Tensor): [batch_size, t_x, t_y]
path = maximum_path(value, mask, implementation="cython")
```

The `implementation` argument allows choosing from one of the following
implementations:

- `cython` (default): Cython-optimised
- `numpy`: pure Numpy

## References

This implementation is taken from the original [Glow-TTS
repository](https://github.com/jaywalnut310/glow-tts). Consider citing the
Glow-TTS paper when using this project:

```bibtex
@inproceedings{kim2020_glowtts,
    title={Glow-{TTS}: A Generative Flow for Text-to-Speech via Monotonic Alignment Search},
    author={Jaehyeon Kim and Sungwon Kim and Jungil Kong and Sungroh Yoon},
    booktitle={Proceedings of Neur{IPS}},
    year={2020},
}
```
