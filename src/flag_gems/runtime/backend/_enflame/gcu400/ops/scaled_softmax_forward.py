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

"""GCU400 scaled softmax forward: ``softmax(input * scale_factor, dim=-1)``.

输入布局 (batch, heads, query, key) 且 key 维为最后一维，连续张量可直接视为
(M=b*h*q, N=k) 的按行 softmax，所以这里复用 enflame softmax 前向的行式数据流：

* 行长 N 能被单个 BLOCK_N 覆盖时，一行（或多行）只占一个 tile，输入只读一遍、
  输出写一遍（寄存器内完成 max/exp/sum），这是 64~4096 这些测试形状的主路径；
* 行长超过 tile 预算时退化为两遍：第一遍 online 求 max/sum，第二遍重算 exp 并写回。

program 总数固定为 ``GRID_DIM_FWD``，每个 program 用 ``tl.range`` 跨步处理多行，
避免行数决定 CTA 数导致的启动与调度开销。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# With 24 SiPs on GCU400, 48 programs is the measured bandwidth sweet spot (same criterion as the softmax forward).
GRID_DIM_FWD = 48
# Element budget for a single tile: BLOCK_M * BLOCK_N must not exceed it.
# Measured on GCU400 (fp16, do_bench): 65536 elements/block is 1.3~1.45x faster than 32768
# ((16,16,1024,1024) 0.81ms vs 1.18ms), while 131072 fails to compile due to insufficient DSM.
ROW_TILE_ELEMS = 65536
# Maximum number of columns covered by a single segment within a row, to avoid overly long rows blowing up the reduction
# width.
MAX_BLOCK_N = 4096
# exp is implemented with exp2, requiring multiplication by log2(e); it must be instantiated as a constexpr to be
# referenced inside the kernel.
LOG2E = tl.constexpr(1.4426950408889634)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_forward_row_kernel(
    out_ptr,
    in_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """N 能被单个 BLOCK_N 覆盖：整行一个 tile，单遍读 + 单遍写。"""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < M) & n_mask[None, :]
        offsets = m_offsets[:, None] * N + n_offsets[None, :]

        # Perform scaling in the fp32 domain to avoid extra rounding from multiplying scale in fp16.
        inp = tl.load(in_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        inp = inp * scale

        row_max = tl.max(inp, 1)
        exp_val = tl.exp2((inp - row_max[:, None]) * LOG2E)
        row_sum = tl.sum(exp_val, 1)
        out = exp_val * (1.0 / row_sum)[:, None]

        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "scale"])
def scaled_softmax_forward_multipass_kernel(
    out_ptr,
    in_ptr,
    scale,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """行长超过 BLOCK_N：第一遍 online 求 max/sum，第二遍重算并写出。"""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    n_offsets = tl.arange(0, BLOCK_N)

    for m_start in tl.range(pid * BLOCK_M, M, num_programs * BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < M
        row_base = m_offsets[:, None] * N

        row_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        row_sum = tl.zeros([BLOCK_M], tl.float32)

        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(
                in_ptr + row_base + n_index[None, :], mask=mask, other=-float("inf")
            )
            inp = inp.to(tl.float32) * scale

            m_new = tl.maximum(row_max, tl.max(inp, 1))
            # When the entire row is -inf, keep the state to avoid NaN from -inf - (-inf).
            all_neg_inf = m_new == float("-inf")
            shifted = tl.sum(tl.exp2((inp - m_new[:, None]) * LOG2E), 1)
            row_sum = tl.where(
                all_neg_inf,
                row_sum,
                row_sum * tl.exp2((row_max - m_new) * LOG2E) + shifted,
            )
            row_max = m_new

        inv_sum = 1.0 / row_sum
        for n_start in tl.range(0, N, BLOCK_N):
            n_index = n_start + n_offsets
            mask = m_mask[:, None] & (n_index[None, :] < N)
            inp = tl.load(
                in_ptr + row_base + n_index[None, :], mask=mask, other=-float("inf")
            ).to(tl.float32)
            exp_val = tl.exp2((inp * scale - row_max[:, None]) * LOG2E)
            out = exp_val * inv_sum[:, None]
            tl.store(
                out_ptr + row_base + n_index[None, :],
                out.to(out_ptr.dtype.element_ty),
                mask=mask,
            )


def _grid_dim_for(total_elems):
    """按总元素数选择常驻 program 数。

    GCU 上 program 启动开销明显（实测 (1,4,64,64) 用 256 个 program 要 25us，
    16~48 个只要 5.8~8.1us），因此小规模问题反而要收窄网格。
    """
    if total_elems <= 65536:
        return 16
    if total_elems <= 262144:
        return 32
    return GRID_DIM_FWD


def _tile_shape(num_rows, row_length, grid_dim):
    """按「行数 / 网格宽度」与 tile 元素预算推导 (BLOCK_M, BLOCK_N)。

    BLOCK_M 必须是 2 的幂（kernel 内 tl.arange 的要求），所以取不超过
    cdiv(num_rows, grid_dim) 的最大 2 的幂，保证 grid_dim 个 program 都能被填满。
    """
    block_n = min(triton.next_power_of_2(row_length), MAX_BLOCK_N)
    target_rows = max(1, triton.cdiv(num_rows, grid_dim))
    block_m = 1 << (target_rows.bit_length() - 1)
    block_m = min(block_m, ROW_TILE_ELEMS // block_n, triton.next_power_of_2(num_rows))
    return max(1, block_m), block_n


def scaled_softmax_forward(input_t: torch.Tensor, scale_factor: float):
    logger.debug("GEMS_ENFLAME SCALED_SOFTMAX_FORWARD")
    assert input_t.dim() == 4, "expected 4D tensor"
    assert input_t.dtype in [
        torch.float16,
        torch.bfloat16,
    ], "Only fp16 and bf16 are supported"

    batch_size, attn_heads, query_seq_len, key_seq_len = input_t.shape
    assert key_seq_len <= 16384, "Key sequence length must be 16384 or less"
    assert key_seq_len % 8 == 0, "Key sequence length must be divisible by 8"
    assert query_seq_len > 1, "Query sequence length must be greater than 1"

    # The row-based data flow requires the last dimension to be contiguous; when (b,h,q,k) is contiguous, the row index
    # is simply (b*H+h)*Q+q and can be flattened directly.
    src = input_t if input_t.is_contiguous() else input_t.contiguous()
    output_t = torch.empty_like(src)

    num_rows = batch_size * attn_heads * query_seq_len
    total_elems = num_rows * key_seq_len
    grid_dim = _grid_dim_for(total_elems)
    block_m, block_n = _tile_shape(num_rows, key_seq_len, grid_dim)
    num_programs = min(grid_dim, triton.cdiv(num_rows, block_m))

    if key_seq_len <= block_n:
        scaled_softmax_forward_row_kernel[(num_programs,)](
            output_t,
            src,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    else:
        scaled_softmax_forward_multipass_kernel[(num_programs,)](
            output_t,
            src,
            scale_factor,
            num_rows,
            key_seq_len,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=4,
        )
    return output_t
