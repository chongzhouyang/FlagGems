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

"""RMSNorm with W8A16 INT8 weight quantization (enflame GCU400).

权重按 ``group_size`` 分组做 INT8 对称量化，kernel 内即时反量化：

    w[n] = weight_q[n] * weight_scale[n // group_size]
    y    = x * rsqrt(mean(x^2) + eps) * w

行批（``ROWS`` 行 x ``BLOCK_N`` 列的 2-D tile）形态：每 program 用一次
[ROWS, BLOCK_N] tile 归一化多行，权重的 INT8 量化值与分组 scale 是循环不变量，
每个 program 只装载一次。逐行（1 program/行）形态下这两份权重对每一行都要
重新读一遍——(1024, 4096) 上权重读写就占了总流量的 1/3，且 M 个 CTA 的
retire/重发射间隙无法被摊薄。

行宽超过单 tile 预算时退化为「分块累加方差 -> 分块写回」两趟；残行尾块走
1-D 行体（2-D masked 尾块是本 backend 的静默错值区）。

三个针对 GCU400 的写法约束：

1. **组号用移位而不是整除**：``weight_scale`` 的下标是逐元素的
   ``col // group_size``。GCU 没有整数除法硬件，4096 宽向量上的 ``//`` 会展开成
   逐 lane 的长序列；``group_size`` 是 2 的幂时改用 ``col >> log2(group_size)``
   （默认 128 即 ``>> 7``），结果逐位相同。
2. **满 tile 不加 mask**：``N`` 恰好整除 ``BLOCK_N`` 时 ``col < N`` 恒真，带 mask
   的 load/store 会退化成逐 lane 谓词写。用 ``EXACT_TILE`` 常量分支把 mask 去掉，
   非整除形状仍走原 masked 路径。
3. **权重按 [1, BLOCK_N] 2-D 形状装载**：1-D 向量广播操作数在 BLOCK 512/1024 +
   ``num_warps >= 2`` 是已验证的静默错值窗，2-D 形状形态在本 backend 的行批
   kernel 上全触发窗验证正确。
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# Upper limit of single-tile column width: when the row width <= 16384, use one tile for the whole row (single pass);
# split into chunks only when wider.
# Measured: for (64,16384), a whole-row tile is 26% faster than two passes of 4096 chunks; for (64,8192), 39% faster.
_MAX_BLOCK_N = 16384
# Row-batch tile element budget (ROWS * BLOCK_N, about 131KB in fp32 live-set terms): R >= 32 with
# tile >= 32768 is a triton_gcu silent-wrong-result region, so R is capped at 8.
_TILE_ELEM_MAX = 32768
_ROWS_CAP = 8
# When M is too small to fill one wave (<= one CTAS width), row-batch brings no benefit and the row-by-row kernel is
# used.
_MIN_ROW_BATCH_M = 48
_TARGET_CTAS = 48
_NUM_WARPS = 2  # Pure 2-D band with no tail branch
_TAIL_NUM_WARPS = (
    1  # When the 1-D tail branch is compiled into the kernel, nw=1 is required
)
_SINGLE_ROW_NUM_WARPS = (
    4  # rows == 1 (very small M) band: intra-CTA parallelism is more valuable
)


@triton.jit
def _group_index(cols, GROUP_SIZE: tl.constexpr, LOG_GROUP: tl.constexpr):
    """组号：2 的幂时移位，否则退回整除（GCU 上整除逐 lane 展开，很贵）。"""
    if LOG_GROUP > 0:
        return cols >> LOG_GROUP
    else:
        return cols // GROUP_SIZE


@triton.jit
def _w8a16_rows_tile(
    Y,
    X,
    WQ,
    WS,
    wq,
    ws,
    row0,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT_TILE: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LOG_GROUP: tl.constexpr,
):
    """整 [ROWS, BLOCK_N] tile 体；wq/ws 是已反量化好的 [1, BLOCK_N] 2-D 权重。"""
    rows = row0 + tl.arange(0, ROWS)
    if SINGLE_CHUNK:
        # Row width is exactly one tile: the x tile stays in registers, completed in one pass.
        cols = tl.arange(0, BLOCK_N)
        ptrs = X + rows[:, None] * N + cols[None, :]
        if EXACT_TILE:
            x = tl.load(ptrs).to(tl.float32)
        else:
            col_mask = cols < N
            x = tl.load(ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rrms = tl.math.rsqrt(var + eps)
        # The wq passed in by the caller has already been multiplied by the group scale (wq * ws), so no dequantization
        # is repeated here.
        y = x * rrms[:, None] * wq
        y_ptrs = Y + rows[:, None] * N + cols[None, :]
        if EXACT_TILE:
            tl.store(y_ptrs, y.to(Y.dtype.element_ty))
        else:
            tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=col_mask[None, :])
    else:
        # Wide rows: pass 1 only accumulates the sum of squares (x tile not resident), pass 2 reloads and writes back.
        acc = tl.zeros([ROWS, BLOCK_N], dtype=tl.float32)
        for off in tl.range(0, N, BLOCK_N, num_stages=2):
            cols = off + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            ptrs = X + rows[:, None] * N + cols[None, :]
            if EXACT_TILE:
                x = tl.load(ptrs).to(tl.float32)
            else:
                x = tl.load(ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
            acc += x * x
        rrms = tl.math.rsqrt(tl.sum(acc, axis=1) / N + eps)
        for off in tl.range(0, N, BLOCK_N, num_stages=2):
            cols = off + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            c2 = cols[None, :]
            if EXACT_TILE:
                x = tl.load(X + rows[:, None] * N + c2).to(tl.float32)
                wq_c = tl.load(WQ + cols)[None, :].to(tl.float32)
                ws_c = tl.load(WS + _group_index(cols, GROUP_SIZE, LOG_GROUP))[
                    None, :
                ].to(tl.float32)
            else:
                x = tl.load(
                    X + rows[:, None] * N + c2, mask=col_mask[None, :], other=0.0
                ).to(tl.float32)
                wq_c = tl.load(WQ + cols, mask=col_mask, other=0)[None, :].to(
                    tl.float32
                )
                ws_c = tl.load(
                    WS + _group_index(cols, GROUP_SIZE, LOG_GROUP),
                    mask=col_mask,
                    other=0.0,
                )[None, :].to(tl.float32)
            y = x * rrms[:, None] * wq_c * ws_c
            y_ptrs = Y + rows[:, None] * N + c2
            if EXACT_TILE:
                tl.store(y_ptrs, y.to(Y.dtype.element_ty))
            else:
                tl.store(y_ptrs, y.to(Y.dtype.element_ty), mask=col_mask[None, :])


@triton.jit
def _w8a16_row(
    Y,
    X,
    WQ,
    WS,
    row,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    EXACT_TILE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LOG_GROUP: tl.constexpr,
):
    """1-D 单行体（尾块回退，任意行宽下均正确）。"""
    Y += row * N
    X += row * N
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        if EXACT_TILE:
            x = tl.load(X + cols).to(tl.float32)
        else:
            x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        acc += x * x
    rrms = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        idx = _group_index(cols, GROUP_SIZE, LOG_GROUP)
        if EXACT_TILE:
            x = tl.load(X + cols).to(tl.float32)
            wq = tl.load(WQ + cols).to(tl.float32)
            ws = tl.load(WS + idx).to(tl.float32)
            tl.store(Y + cols, (x * rrms * wq * ws).to(Y.dtype.element_ty))
        else:
            mask = cols < N
            x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
            wq = tl.load(WQ + cols, mask=mask, other=0).to(tl.float32)
            ws = tl.load(WS + idx, mask=mask, other=0.0).to(tl.float32)
            tl.store(Y + cols, (x * rrms * wq * ws).to(Y.dtype.element_ty), mask=mask)


@triton.jit(
    do_not_specialize=["eps"],
)
def _rms_norm_w8a16_rows_kernel(
    Y,
    X,
    WQ,
    WS,
    M,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT_TILE: tl.constexpr,
    SINGLE_CHUNK: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    """每 program 经 [ROWS, BLOCK_N] 2-D tile 归一化 ROWS 行。"""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS)
    cols = tl.arange(0, BLOCK_N)
    # The dequantized weight is loop-invariant: each program loads it once in [1, BLOCK_N] 2-D shape
    # (dequantization is placed outside the loop to avoid recomputing wq * ws every row).
    if EXACT_TILE:
        wq = tl.load(WQ + cols[None, :]).to(tl.float32)
        ws = tl.load(WS + _group_index(cols, GROUP_SIZE, LOG_GROUP)[None, :]).to(
            tl.float32
        )
    else:
        col_mask = cols < N
        wq = tl.load(WQ + cols[None, :], mask=col_mask[None, :], other=0).to(tl.float32)
        ws = tl.load(
            WS + _group_index(cols, GROUP_SIZE, LOG_GROUP)[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
    wq = wq * ws

    if HAS_TAIL:
        # 1-D tail branch compiled into the kernel: must launch with num_warps=1.
        for tile in range(pid, num_tiles, num_programs):
            row0 = tile * ROWS
            if row0 + ROWS <= M:
                _w8a16_rows_tile(
                    Y,
                    X,
                    WQ,
                    WS,
                    wq,
                    ws,
                    row0,
                    N,
                    eps,
                    BLOCK_N,
                    ROWS,
                    EXACT_TILE,
                    SINGLE_CHUNK,
                    GROUP_SIZE,
                    LOG_GROUP,
                )
            else:
                for i in tl.static_range(ROWS):
                    row = row0 + i
                    if row < M:
                        _w8a16_row(
                            Y,
                            X,
                            WQ,
                            WS,
                            row,
                            N,
                            eps,
                            BLOCK_N,
                            EXACT_TILE,
                            GROUP_SIZE,
                            LOG_GROUP,
                        )
    else:
        # The host guarantees M % ROWS == 0: the loop body has no branches, so software-pipeline this loop.
        for tile in tl.range(pid, num_tiles, num_programs, num_stages=2):
            row0 = tile * ROWS
            _w8a16_rows_tile(
                Y,
                X,
                WQ,
                WS,
                wq,
                ws,
                row0,
                N,
                eps,
                BLOCK_N,
                ROWS,
                EXACT_TILE,
                SINGLE_CHUNK,
                GROUP_SIZE,
                LOG_GROUP,
            )


@triton.jit(
    do_not_specialize=["eps", "N"],
)
def _w8a16_per_row_kernel(
    Y,
    X,
    WQ,
    WS,
    N,
    eps,
    GROUP_SIZE: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    EXACT_TILE: tl.constexpr,
):
    """一个 program 处理一行（M 小到铺不满一个 wave 时用，行批无收益）。"""
    row = tl.program_id(0)
    x_row = X + row * N
    y_row = Y + row * N
    cols = tl.arange(0, BLOCK_N)

    if SINGLE_BLOCK and EXACT_TILE:
        # The whole row is exactly one tile: no mask, load/store take the full-width path.
        x = tl.load(x_row + cols).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        wq = tl.load(WQ + cols)
        ws = tl.load(WS + _group_index(cols, GROUP_SIZE, LOG_GROUP)).to(tl.float32)
        y = x * rrms * wq.to(tl.float32) * ws
        tl.store(y_row + cols, y.to(Y.dtype.element_ty))
    elif SINGLE_BLOCK:
        mask = cols < N
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        wq = tl.load(WQ + cols, mask=mask, other=0)
        ws = tl.load(
            WS + _group_index(cols, GROUP_SIZE, LOG_GROUP), mask=mask, other=0.0
        ).to(tl.float32)
        y = x * rrms * wq.to(tl.float32) * ws
        tl.store(y_row + cols, y.to(Y.dtype.element_ty), mask=mask)
    else:
        # Wide rows: pass 1 only accumulates the sum of squares (x tile not resident), pass 2 reloads and writes back.
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for off in range(0, N, BLOCK_N):
            c = off + cols
            mask = c < N
            x = tl.load(x_row + c, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        rrms = tl.math.rsqrt(tl.sum(acc, axis=0) / N + eps)
        for off in range(0, N, BLOCK_N):
            c = off + cols
            mask = c < N
            x = tl.load(x_row + c, mask=mask, other=0.0).to(tl.float32)
            wq = tl.load(WQ + c, mask=mask, other=0)
            ws = tl.load(
                WS + _group_index(c, GROUP_SIZE, LOG_GROUP), mask=mask, other=0.0
            ).to(tl.float32)
            y = x * rrms * wq.to(tl.float32) * ws
            tl.store(y_row + c, y.to(Y.dtype.element_ty), mask=mask)


def _log2_if_pow2(value):
    """2 的幂返回 log2，否则返回 0（kernel 侧据此选择移位或整除）。"""
    if value <= 0 or (value & (value - 1)) != 0:
        return 0
    return value.bit_length() - 1


def _rows_per_cta(m, block_n):
    """行批因子 R：尽量吃满 tile 预算（权重摊薄 + 单趟搬运），受 R 上限约束。"""
    if m <= _MIN_ROW_BATCH_M:
        return 1
    tile_cap = max(1, _TILE_ELEM_MAX // block_n)
    return max(1, min(_ROWS_CAP, tile_cap, m))


def rms_norm_w8a16_int8(
    x, normalized_shape, weight_q, weight_scale, eps=1e-5, group_size=128
):
    logger.debug("GEMS RMS_NORM_W8A16_INT8")
    dim = x.ndim - len(normalized_shape)
    M = math.prod(x.shape[:dim])
    N = math.prod(normalized_shape)
    # The kernel addresses with "row stride == N, last-dim stride == 1"; weights/group scale are also read with compact
    # 1-D indexing; non-contiguous inputs are normalized first (contiguous inputs are zero-copy).
    x = x.contiguous()
    weight_q = weight_q.contiguous()
    weight_scale = weight_scale.contiguous()
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    if M == 0 or N == 0:
        return y

    block_n = min(triton.next_power_of_2(N), _MAX_BLOCK_N)
    rows = _rows_per_cta(M, block_n)
    if rows == 1:
        # M is too small to fill one wave: row-batch has no benefit, keep the row-by-row 1-D kernel.
        with torch_device_fn.device(x.device):
            _w8a16_per_row_kernel[(M,)](
                y,
                x,
                weight_q,
                weight_scale,
                N,
                eps,
                GROUP_SIZE=group_size,
                LOG_GROUP=_log2_if_pow2(group_size),
                BLOCK_N=block_n,
                SINGLE_BLOCK=block_n >= N,
                EXACT_TILE=(N % block_n) == 0,
                num_warps=_SINGLE_ROW_NUM_WARPS,
            )
        return y

    has_tail = (M % rows) != 0
    grid = min(triton.cdiv(M, rows), _TARGET_CTAS)
    num_warps = _TAIL_NUM_WARPS if has_tail else _NUM_WARPS
    with torch_device_fn.device(x.device):
        _rms_norm_w8a16_rows_kernel[(grid,)](
            y,
            x,
            weight_q,
            weight_scale,
            M,
            N,
            eps,
            GROUP_SIZE=group_size,
            LOG_GROUP=_log2_if_pow2(group_size),
            BLOCK_N=block_n,
            ROWS=rows,
            EXACT_TILE=(N % block_n) == 0,
            SINGLE_CHUNK=(block_n >= N),
            HAS_TAIL=has_tail,
            num_warps=num_warps,
        )
    return y
