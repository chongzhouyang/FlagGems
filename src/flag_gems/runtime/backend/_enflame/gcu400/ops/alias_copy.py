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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# On GCU400 one SIP can host two resident programs, so 48 CTAs is full occupancy; more CTAs bring no benefit.
NUM_SIPS = 24
MAX_CTAS = NUM_SIPS * 2

# The bytes moved per CTA per transfer determine DMA efficiency: the 16KB setting is about 24% slower than the 64KB one,
# so derive BLOCK from the element
# width. The cap is a 64KB tile: with two programs/SP, 2×2×64KB = 256KB is still within
# the 448KB on-chip budget; going up to a 128KB tile exceeds the budget and triggers a device abort in the full suite.
DMA_TILE_BYTES = 64 * 1024
MAX_LARGE_BLOCK = 32768
# Small tensors are launch-bound; use a smaller BLOCK in exchange for more CTA parallelism.
SMALL_INPUT_ELEMENTS = 1 << 20
SMALL_BLOCK = 16384
SMALL_NUM_WARPS = 4
TINY_INPUT_ELEMENTS = 4096
TINY_BLOCK = 4096
LARGE_NUM_WARPS = 2


@libentry()
@triton.jit(do_not_specialize=["n_elements", "n_full_blocks"])
def _alias_copy_kernel(
    src_ptr, dst_ptr, n_elements, n_full_blocks, BLOCK: tl.constexpr
):
    """纯字节搬运：整块无掩码 + 尾部单 CTA 带掩码收尾。

    主体循环不带 mask，避免每次 load 的谓词开销；只有不足一个 BLOCK 的残余
    才走掩码路径，且固定由 0 号 CTA 处理。
    """
    program_id = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    lane_offset = tl.arange(0, BLOCK)

    for block_index in tl.range(program_id, n_full_blocks, num_ctas):
        element_offset = block_index * BLOCK + lane_offset
        tl.store(dst_ptr + element_offset, tl.load(src_ptr + element_offset))

    tail_start = n_full_blocks * BLOCK
    if tail_start < n_elements:
        if program_id == 0:
            tail_offset = tail_start + lane_offset
            tail_mask = tail_offset < n_elements
            tl.store(
                dst_ptr + tail_offset,
                tl.load(src_ptr + tail_offset, mask=tail_mask),
                mask=tail_mask,
            )


def _pick_block(n_elements: int, element_size: int):
    """按元素数 / 元素宽度选择 (BLOCK, num_warps)。"""
    if n_elements <= TINY_INPUT_ELEMENTS:
        return TINY_BLOCK, SMALL_NUM_WARPS
    if n_elements <= SMALL_INPUT_ELEMENTS:
        return SMALL_BLOCK, SMALL_NUM_WARPS
    # Large tensors: moving 128KB per CTA per transfer minimizes DMA setup overhead; the width cap prevents an
    # 8B-element
    # tile from exceeding the DSM budget.
    block = min(DMA_TILE_BYTES // max(element_size, 1), MAX_LARGE_BLOCK)
    return block, LARGE_NUM_WARPS


def _launch_copy(src: torch.Tensor, dst: torch.Tensor, n_elements: int):
    block, num_warps = _pick_block(n_elements, src.element_size())
    n_full_blocks = n_elements // block
    grid = min(max(n_full_blocks, 1), MAX_CTAS)
    with torch_device_fn.device(src.device):
        _alias_copy_kernel[(grid,)](
            src,
            dst,
            n_elements,
            n_full_blocks,
            BLOCK=block,
            num_warps=num_warps,
        )


def alias_copy(x: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY")
    out = torch.empty_like(x)
    n_elements = out.numel()
    if n_elements == 0:
        return out
    src = x.contiguous() if not x.is_contiguous() else x
    if not out.is_contiguous():
        out = out.contiguous()
    if src.dtype != out.dtype:
        raise RuntimeError("alias_copy: dtype mismatch between input and output.")
    _launch_copy(src, out, n_elements)
    return out


def alias_copy_out(x: torch.Tensor, out: torch.Tensor):
    logger.debug("GEMS_ENFLAME ALIAS_COPY_OUT")
    if x.dtype != out.dtype:
        raise RuntimeError("alias_copy_out: dtype of input and output must match.")
    if x.numel() != out.numel():
        raise RuntimeError(
            "alias_copy_out: input and output must have the same number of elements."
        )
    if x.device != out.device:
        raise RuntimeError(
            "alias_copy_out: input and output must be on the same device."
        )
    if not out.is_contiguous():
        raise RuntimeError("alias_copy_out: output tensor must be contiguous.")
    src = x.contiguous() if not x.is_contiguous() else x
    n_elements = out.numel()
    if n_elements == 0:
        return out
    _launch_copy(src, out, n_elements)
    return out
