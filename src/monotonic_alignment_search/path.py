# SPDX-FileCopyrightText: Jaehyeon Kim
# SPDX-FileContributor: Jaehyeon Kim
# SPDX-FileContributor: Enno Hermann
#
# SPDX-License-Identifier: MIT

"""Different implementations to find the most likely monotonic alignment.

- Pure Numpy
- Cython optimised
"""

from typing import Any, Literal

import numpy as np
import torch

from monotonic_alignment_search.core import maximum_path_c


def maximum_path_cython(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cython optimised version.

    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]

    Source:
    https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/monotonic_align/__init__.py
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


def maximum_path_numpy(
    value: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_neg_val: float = -np.inf,
) -> torch.Tensor:
    """Numpy-only version.

    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]

    Source:
    https://github.com/jaywalnut310/glow-tts/blob/13e997689d643410f5d9f1f9a73877ae85e19bc2/commons.py#L60
    """
    value = value * mask

    device = value.device
    dtype = value.dtype
    value_np = value.cpu().detach().numpy()
    mask_np = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value_np.shape
    direction = np.zeros(value_np.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[
            :,
            :-1,
        ]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value_np[:, :, j], max_neg_val)
    direction = np.where(mask_np, direction, 1)

    path = np.zeros(value_np.shape, dtype=np.float32)
    index = mask_np[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask_np.astype(np.float32)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def maximum_path(
    value: torch.Tensor,
    mask: torch.Tensor,
    implementation: Literal["cython", "numpy"] = "cython",
    **kwargs: Any,  # noqa: ANN401
) -> torch.Tensor:
    """Get most probably monotonic alignment.

    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    implementation: Cython, pure Numpy
    """
    if implementation == "cython":
        return maximum_path_cython(value, mask)
    if implementation == "numpy":
        return maximum_path_numpy(value, mask, **kwargs)
    raise NotImplementedError
