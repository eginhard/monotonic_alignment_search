<!--
SPDX-FileCopyrightText: Enno Hermann

SPDX-License-Identifier: MIT
-->

# Monotonic Alignment Search (MAS)

Implementation of MAS from [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
for easy reuse in other projects.

## Usage

MAS can find the most probable alignment between a text sequence `t_x` and a
speech sequence `t_y`.

```python
from monotonic_alignment_search import maximum_path

# value (torch.Tensor): [batch_size, t_x, t_y]
# mask  (torch.Tensor): [batch_size, t_x, t_y]
path = maximum_path(value, mask)
```

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
