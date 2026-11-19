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

NUM_SIPS = 24

# Chunking the output side by 4096 elements with one program per SP (grid=NUM_SIPS) is measured best on this card:
# After reducing grid from 4*NUM_SIPS to NUM_SIPS, all shapes are faster for fp16/fp32 (see optimization report R1/R2).
BLOCK = 4096

# R3 重采样（2026-11-19）：1GB 纯拷贝 roofline 显示 grid=48/BLOCK=4096 达 2330GB/s
# （grid=24 仅 1479GB/s）；用 self_int kernel 本体 A/B，grid=48 在全部 benchmark 形状
# （fp16/fp32/bf16，2D-4D）比 grid=24 快 9%~39%，96 开始回落。self_int 系列改用 48；
# NUM_SIPS=24 仅保留给 repeat_interleave_tensor（碎行小 kernel）。
SELF_INT_GRID = 48


@libentry()
@triton.jit
def repeat_interleave_kernel(
    x_ptr,
    out_ptr,
    N_total,
    inner_size,
    repeat_inner: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SLAB: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Each program claims a contiguous slab (blocks are contiguous on the output), improving DRAM page locality
    pid = tl.program_id(0)
    arange = tl.arange(0, BLOCK)
    start = pid * SLAB
    end = min(start + SLAB, NUM_BLOCKS)

    for block_id in tl.range(start, end):
        off = block_id * BLOCK + arange

        inner_idx = off % inner_size
        outer_idx = off // repeat_inner
        src_idx = outer_idx * inner_size + inner_idx

        if MASKED:
            mask = off < N_total
            vals = tl.load(x_ptr + src_idx, mask=mask)
            tl.store(out_ptr + off, vals, mask=mask)
        else:
            # When the last block is fully valid, skip the predicate; measured about 3% gain for fp16
            vals = tl.load(x_ptr + src_idx)
            tl.store(out_ptr + off, vals)


@libentry()
@triton.jit
def repeat_interleave_flat_kernel(
    x_ptr,
    out_ptr,
    N_total,
    repeats: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    SLAB: tl.constexpr,
    MASKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    arange = tl.arange(0, BLOCK)
    start = pid * SLAB
    end = min(start + SLAB, NUM_BLOCKS)

    for block_id in tl.range(start, end):
        off = block_id * BLOCK + arange
        src_idx = off // repeats

        if MASKED:
            mask = off < N_total
            vals = tl.load(x_ptr + src_idx, mask=mask)
            tl.store(out_ptr + off, vals, mask=mask)
        else:
            vals = tl.load(x_ptr + src_idx)
            tl.store(out_ptr + off, vals)


def repeat_interleave_self_int(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_SELF_INT")
    if dim is None:
        inp = inp.contiguous().flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
        if dim < 0:
            dim = dim + inp.ndim

    inp_shape = list(inp.shape)
    output_shape = list(inp.shape)
    output_shape[dim] *= repeats

    if output_size is not None and output_size != output_shape[dim]:
        raise RuntimeError(
            "repeat_interleave: Invalid output_size, expected {} but got {}".format(
                output_shape[dim], output_size
            )
        )

    output = torch.empty(output_shape, dtype=inp.dtype, device=inp.device)

    if repeats == 0:
        return output

    inp = inp.contiguous()
    N_total = output.numel()

    NUM_BLOCKS = triton.cdiv(N_total, BLOCK)
    grid_size = min(NUM_BLOCKS, SELF_INT_GRID)
    slab = triton.cdiv(NUM_BLOCKS, grid_size)
    masked = (N_total % BLOCK) != 0

    inner_size = 1
    for i in range(dim + 1, len(inp_shape)):
        inner_size *= inp_shape[i]

    with torch_device_fn.device(inp.device):
        if inner_size == 1:
            repeat_interleave_flat_kernel[(grid_size,)](
                inp,
                output,
                N_total,
                repeats=repeats,
                NUM_BLOCKS=NUM_BLOCKS,
                SLAB=slab,
                MASKED=masked,
                BLOCK=BLOCK,
            )
        else:
            repeat_inner = repeats * inner_size
            repeat_interleave_kernel[(grid_size,)](
                inp,
                output,
                N_total,
                inner_size,
                repeat_inner=repeat_inner,
                NUM_BLOCKS=NUM_BLOCKS,
                SLAB=slab,
                MASKED=masked,
                BLOCK=BLOCK,
            )

    return output


@libentry()
@triton.jit
def repeat_interleave_tensor_kernel(
    repeats_ptr, cumsum_ptr, out_ptr, size, BLOCK_SIZE: tl.constexpr
):
    # Fixed-grid friendly: stride over repeat indices so the launch never
    # exceeds the 65535 grid.x limit for large repeat vectors (grid used to
    # be (size,), which overflows at size=65536).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    for rid in tl.range(pid, size, num_pids):
        cumsum = tl.load(cumsum_ptr + rid)
        repeats = tl.load(repeats_ptr + rid)
        # Local base (do NOT advance out_ptr in-place: the stride loop would
        # accumulate the offset across iterations).
        out_base = out_ptr + cumsum - repeats

        tl.device_assert(repeats >= 0, "repeats can not be negative")

        for start_k in range(0, repeats, BLOCK_SIZE):
            offsets_k = start_k + tl.arange(0, BLOCK_SIZE)
            mask_k = offsets_k < repeats
            tl.store(out_base + offsets_k, rid, mask=mask_k)


@libentry()
@triton.jit
def repeat_interleave_gather_kernel(
    inp_ptr,
    out_ptr,
    repeats_ptr,
    cumsum_ptr,
    N,
    inner_size,
    BLOCK: tl.constexpr,
    EVEN_INNER: tl.constexpr,
):
    # Fused dim==0 path for repeat_interleave.self_Tensor: one kernel replaces
    # the materialize-indices + index_select pair. 2D grid over (input row i,
    # inner chunk c): each CTA loads its source chunk ONCE and stores it to
    # every output row of i's run, so source read traffic is N*inner instead
    # of total_rows*inner (the run length is repeats[i], a runtime loop bound
    # that skips repeats==0 rows for free). No cross-warp reduction anywhere:
    # a per-row cumsum search measured ~2us/row on this card, which dominated
    # every small-inner shape.
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    num_x = tl.num_programs(0)
    num_y = tl.num_programs(1)
    offs = tl.arange(0, BLOCK)
    inner_blocks = tl.cdiv(inner_size, BLOCK)
    for c in tl.range(pid_y, inner_blocks, num_y):
        coff = c * BLOCK + offs
        for i in tl.range(pid_x, N, num_x):
            # int32 domain is guaranteed by the host-side numel guards
            repeats_i = tl.load(repeats_ptr + i).to(tl.int32)
            cumsum_i = tl.load(cumsum_ptr + i).to(tl.int32)
            base = cumsum_i - repeats_i
            if EVEN_INNER:
                vals = tl.load(inp_ptr + i * inner_size + coff)
                for k in tl.range(0, repeats_i):
                    tl.store(out_ptr + (base + k) * inner_size + coff, vals)
            else:
                cmask = coff < inner_size
                vals = tl.load(inp_ptr + i * inner_size + coff, mask=cmask)
                for k in tl.range(0, repeats_i):
                    tl.store(out_ptr + (base + k) * inner_size + coff, vals, mask=cmask)


# 纯拷贝 roofline 实测（1GB int32, r+w 2330GB/s）：grid=48、BLOCK=4096 附近最优；
# gather kernel 的 x 维沿用该采样点。y 维受 GCU_MAX_GRID_Y 硬上限约束，跨步循环覆盖。
GATHER_GRID_X = 48
GCU_MAX_GRID_Y = 63
# 超大 inner（>=2^17）单行连续段足够长时，BLOCK=8192 的连续 store 实测再快 ~6%
GATHER_BIG_INNER = 2**17
# 行内元素过小时走组合路径（index_select 对碎行更友好），避免极小 store 粒度
MIN_FUSED_INNER = 16


def repeat_interleave_tensor(repeats, *, output_size=None):
    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_TENSOR")
    assert repeats.ndim == 1, "repeat_interleave only accept 1D vector as repeat"
    cumsum = repeats.cumsum(axis=0)
    result_size = cumsum[-1].item()
    assert result_size >= 0, "repeats can not be negative"
    out = torch.empty((result_size,), dtype=repeats.dtype, device=repeats.device)
    size = repeats.size(0)
    grid = (min(size, NUM_SIPS * 4),)
    BLOCK_SIZE = 32
    repeat_interleave_tensor_kernel[grid](
        repeats,
        cumsum,
        out,
        size,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
    )
    return out


def repeat_interleave_self_tensor(inp, repeats, dim=None, *, output_size=None):
    logger.debug("GEMS_ENFLAME REPEAT_INTERLEAVE_SELF_TENSOR")
    if dim is None:
        inp = inp.flatten()
        dim = 0
    else:
        if dim < -inp.ndim or dim >= inp.ndim:
            raise IndexError(
                "Dimension out of range (expected to be in range of [{}, {}], but got {})".format(
                    -inp.ndim, inp.ndim - 1, dim
                )
            )
    if repeats.ndim == 0 or (repeats.ndim == 1 and repeats.size(0) == 1):
        return repeat_interleave_self_int(
            inp, repeats.item(), dim=dim, output_size=output_size
        )
    elif repeats.ndim > 1:
        raise RuntimeError("repeats must be 0-dim or 1-dim tensor")
    if repeats.numel() == 0:
        return inp.clone()
    inp_shape = list(inp.shape)
    if dim < 0:
        dim = dim + len(inp_shape)
    if repeats.size(0) != inp_shape[dim]:
        raise RuntimeError("repeats must have the same size as input along dim")

    if dim == 0:
        inner_size = 1
        for s in inp_shape[1:]:
            inner_size *= s
        N = inp_shape[0]
        # int32 安全域：本路径 kernel 内偏移/repeats/cumsum 全部按 int32 计；
        # 越界即回退组合路径（int64 索引机制不变）。
        if inner_size >= MIN_FUSED_INNER and inp.numel() <= 0x7FFFFFFF:
            cumsum = repeats.cumsum(axis=0)
            total_rows = int(cumsum[-1].item())
            assert total_rows >= 0, "repeats can not be negative"
            if total_rows * inner_size <= 0x7FFFFFFF:
                out_shape = [total_rows] + inp_shape[1:]
                output = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
                if total_rows == 0:
                    return output
                if not inp.is_contiguous():
                    inp = inp.contiguous()
                if not repeats.is_contiguous():
                    repeats = repeats.contiguous()
                cap = 8192 if inner_size >= GATHER_BIG_INNER else 4096
                block = min(triton.next_power_of_2(inner_size), cap)
                repeat_interleave_gather_kernel[
                    (
                        min(N, GATHER_GRID_X),
                        min(triton.cdiv(inner_size, block), GCU_MAX_GRID_Y),
                    )
                ](
                    inp,
                    output,
                    repeats,
                    cumsum,
                    N,
                    inner_size,
                    BLOCK=block,
                    EVEN_INNER=(inner_size % block) == 0,
                    num_warps=4,
                )
                return output

    indices = repeat_interleave_tensor(repeats)
    res = torch.index_select(inp, dim, indices)
    return res
