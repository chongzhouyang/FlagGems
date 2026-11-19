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

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# GCU grid.x hardware limit (65535): large pointwise tensors must use a
# grid-stride loop instead of a one-block-per-tile launch.
_MAX_GRID = 65535
# Number of SIPS on GCU400 (resident grid cap consistent with other pointwise ops)
_NUM_SIPS = 24


def _round_config(n_elements, itemsize):
    """选 (BLOCK_SIZE, num_warps, grid_cap)。

    该 kernel 是 1 读 1 写的纯访存流，实测上限就是显存带宽（本卡约
    200 GB/s），所以配置完全按实测带宽挑：
    - 16 位：每 tile 只有 8 KB 数据，按 tile 数发几千个 CTA 时波次调度开销
      明显（16.7M 元素只有 157 GB/s）。改成常驻网格（48 个 CTA）+ grid-stride
      循环后能跑到 190 GB/s，超大张量（512M）从 181 GB/s 提到 195 GB/s。
    - 32 位：每 tile 16 KB，按 tile 发 CTA 反而最快（195~201 GB/s），保持原样。
    - 极小张量（n ≤ 4096）：单 CTA 直接覆盖，避免空转。
    """
    if n_elements <= 4096:
        return triton.next_power_of_2(max(n_elements, 1)), 4, _NUM_SIPS * 2
    if itemsize >= 4:
        return 4096, 8, _MAX_GRID
    return 4096, 4, _NUM_SIPS * 2


@libentry()
@triton.jit
def _special_round_kernel(
    x_ptr,
    out_ptr,
    scale,
    recip,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    step = tl.num_programs(0) * BLOCK_SIZE
    for block_start in range(pid * BLOCK_SIZE, n_elements, step):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        # PyTorch uses fp32 for intermediate computation (not fp64).
        # Using fp32 matches PyTorch's behavior where fp32 precision truncation
        # can snap intermediate values to exact ties (e.g. 1.2345 * 1000 = 1234.5).
        x_f32 = x.to(tl.float32) * scale

        # Banker's rounding: round half to even.
        sign = tl.where(x_f32 >= 0.0, 1.0, -1.0)
        abs_x = tl.math.abs(x_f32)
        floor_x = tl.math.floor(abs_x)
        frac = abs_x - floor_x
        is_even = (floor_x % 2.0) == 0.0

        rounded_abs = tl.where(
            frac > 0.5,
            floor_x + 1.0,
            tl.where(
                frac < 0.5,
                floor_x,
                tl.where(is_even, floor_x, floor_x + 1.0),
            ),
        )
        result = sign * rounded_abs
        # Rescaling: GCU's fp32 '/' is, like torch_gcu's native aten::special_round,
        # a reciprocal-multiply (1/10 is inexact in fp32, so 26*0.1f lands on
        # the upper neighbor of 2.6). This must match that semantics bit-for-bit: do not use Markstein FMA
        # to correct the quotient to the mathematically correctly rounded value, otherwise it will be 1 ULP off from the
        # reference and fail
        # (measured on 2026-11-17 with the decimals=1/3 case).
        result = (result * recip).to(x.dtype)

        tl.store(out_ptr + offsets, result, mask=mask)


def _launch(x, out, scale):
    n_elements = out.numel()
    if n_elements == 0:
        return out
    block_size, num_warps, grid_cap = _round_config(n_elements, out.element_size())
    grid = min(triton.cdiv(n_elements, block_size), grid_cap, _MAX_GRID)
    _special_round_kernel[(grid,)](
        x,
        out,
        scale,
        1.0 / scale,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return out


def special_round(A, *, decimals=0):
    logger.debug("GEMS_ENFLAME SPECIAL_ROUND")
    logger.debug("GEMS_ENFLAME SPECIAL_ROUND_OUT")
    scale = float(10**decimals)
    out = torch.empty_like(A)
    return _launch(A, out, scale)


def special_round_out(self, out, *, decimals=0):
    scale = float(10**decimals)
    return _launch(self, out, scale)
