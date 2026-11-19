# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# GCU400 customization (2026-11-17):
# 1) Fix the hard benchmark crash: the official bench calls this op with device='gcu' (no index string),
#    so `inp.device == device` is always False and it falls into generic `_resize_output` -> generic
#    copy_ with a 1D unbounded grid (at the 2^30-element scale grid.x=131072 > the hard limit of 65535, OutOfResources).
#    Device normalization: an index-less device of the same type means "the current device of that type"; for the same
# physical device, go directly
#    through the in-place metadata path; genuinely cross-device cases go through the local enflame `_resize_output`
#    (whose internal copy_ uses a 48-CTA grid-stride loop, so no scale hits the limit).
# 2) The in-place path goes straight to `set_.source_Storage_storage_offset` (FlagGems does not register this
#    overload, so behavior is deterministic in both dispatch states): zero aten calls in the same-shape steady state
# (inductor
#    output-buffer reuse mostly involves repeated shapes; this mirrors native resize_'s same-shape early exit).
#    torch_gcu's set_ has no capacity check - an out-of-bounds set_ silently succeeds and out-of-bounds reads risk
# MMU/RAS errors,
#    so grow performs a (offset+numel)*itemsize capacity check; if insufficient, following native resize_ semantics it
#    allocates new storage + copies the linear prefix (local copy_).
import logging
import math
from typing import List

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops._resize_output import _resize_output
from flag_gems.runtime.backend._enflame.gcu400.ops.copy import copy_

logger = logging.getLogger(__name__)


def _is_same_device(tensor_device: torch.device, device) -> bool:
    """``device``（str / torch.device，可无 index）是否解析到与 ``tensor_device``
    相同的物理设备。无 index 的同类型 device 指该类型当前设备；非 index 型设备
    类型（如 cpu）唯一，直接视为相同。
    """
    if isinstance(device, torch.device):
        target_type, target_index = device.type, device.index
    elif isinstance(device, str):
        if ":" in device:
            try:
                parsed = torch.device(device)
            except ValueError:
                return False
            target_type, target_index = parsed.type, parsed.index
        else:
            target_type, target_index = device, None
    else:
        return False
    if target_type != tensor_device.type:
        return False
    if target_index is None:
        return tensor_device.index is None or (
            tensor_device.index == torch_device_fn.current_device()
        )
    return tensor_device.index == target_index


def _resize_tensor_inplace(inp: torch.Tensor, size) -> torch.Tensor:
    """aten::resize_ 等价原地实现，不经 aten 分发（use_gems 态 resize_ 路由到
    本地 set_，其无容量检查；本实现自行保证不越界）。"""
    new_numel = math.prod(size)
    old_numel = inp.numel()
    storage = inp.untyped_storage()
    offset = inp.storage_offset()
    if (
        new_numel <= old_numel
        or (offset + new_numel) * inp.element_size() <= storage.nbytes()
    ):
        inp.set_(storage, offset, size)
        return inp
    # grow beyond capacity: allocate new storage + copy the linear prefix (native resize_ semantics; for non-contiguous
    # inp, reshape(-1) materializes logical order, consistent with generic _resize_output)
    new = torch.empty(size, dtype=inp.dtype, device=inp.device)
    if old_numel:
        copy_(new.reshape(-1)[:old_numel], inp.reshape(-1))
    inp.set_(new.untyped_storage(), 0, size)
    return inp


def _resize_output_(inp: torch.Tensor, size: List[int], device: torch.device):
    logger.debug("GEMS_ENFLAME _RESIZE_OUTPUT_")
    if _is_same_device(inp.device, device):
        if inp.is_contiguous() and inp.shape == torch.Size(size):
            return inp
        return _resize_tensor_inplace(inp, size)
    out = _resize_output(inp, size, device)
    _resize_tensor_inplace(inp, size)
    inp.copy_(out)
    return inp
