# SPDX-FileCopyrightText: Jaehyeon Kim
#
# SPDX-License-Identifier: MIT

# Source:
# https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/monotonic_align/__init__.py

"""Monotonic alignment search."""

import numpy as np
import torch

from monotonic_alignment_search.core import maximum_path_c


def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cython optimised version.

    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    value = value * mask
    device = value.device
    dtype = value.dtype
    value_np = value.data.cpu().numpy().astype(np.float32)
    path = np.zeros_like(value_np).astype(np.int32)
    mask_np = mask.data.cpu().numpy()

    t_x_max = mask_np.sum(1)[:, 0].astype(np.int32)
    t_y_max = mask_np.sum(2)[:, 0].astype(np.int32)
    maximum_path_c(path, value_np, t_x_max, t_y_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
