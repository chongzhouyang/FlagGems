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

# Semantics (bit-for-bit aligned with ATen's integer gcd; cross-checked on the host with a sim script across all dtypes
# at 8/16/32/64 bits):
#   a = |a|, b = |b| take absolute values **within this dtype**, so negating INT_MIN still wraps around to INT_MIN;
#   then run the signed Euclidean loop while (b) { r = a % b; a = b; b = r; } and directly return the final a,
#   with no final absolute value. So the result is allowed to be negative, and when INT_MIN is involved it is not the
# "absolute value of the true gcd",
#   e.g. gcd(INT32_MIN, 6) == -2, gcd(INT32_MIN, 0) == INT32_MIN.
# The original implementation used binary-GCD(ctz) + an INT_MIN special case to "get the magnitude but not the sign",
# which is
# the root cause of the failures of tests/test_gcd.py on int16/int32 (INT_MIN boundary and a few full-range elements).


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def gcd_kernel_i32(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    DTYPE_MIN: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """8/16/32 位整数统一在 int32 里算：本 dtype 取值范围是 int32 的子集，
    只有 INT_MIN 的取绝对值需要按 dtype 回绕，其余运算结果与 dtype 内运算一致。"""
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (n_elements + BLOCK - 1) // BLOCK, num_pids):
        offsets = block_id * BLOCK + arange
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0).to(tl.int32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0).to(tl.int32)
        a = tl.where(x == DTYPE_MIN, x, tl.abs(x))
        b = tl.where(y == DTYPE_MIN, y, tl.abs(y))

        # Lanes with active=false leave a unchanged (when b==0 from the start, output |a| directly);
        # Perform an intra-block reduction only once every CHUNK iterations, reducing the cross-lane reduction overhead
        # to 1/CHUNK.
        active = mask & (b != 0)
        while tl.sum(active.to(tl.int32), axis=0) > 0:
            for _ in tl.static_range(CHUNK):
                r = a % tl.where(active, b, 1)
                a = tl.where(active, b, a)
                b = tl.where(active, r, b)
                active = active & (b != 0)

        tl.store(out_ptr + offsets, a.to(out_ptr.type.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def gcd_kernel_i64(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    DTYPE_MIN: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    for block_id in tl.range(pid, (n_elements + BLOCK - 1) // BLOCK, num_pids):
        offsets = block_id * BLOCK + arange
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0)
        a = tl.where(x == DTYPE_MIN, x, tl.abs(x))
        b = tl.where(y == DTYPE_MIN, y, tl.abs(y))

        active = mask & (b != 0)
        while tl.sum(active.to(tl.int32), axis=0) > 0:
            for _ in tl.static_range(CHUNK):
                r = a % tl.where(active, b, 1)
                a = tl.where(active, b, a)
                b = tl.where(active, r, b)
                active = active & (b != 0)

        tl.store(out_ptr + offsets, a, mask=mask)


def _kernel_meta(dtype):
    """(kernel, BLOCK, num_warps, CHUNK)。

    BLOCK=4096 / CHUNK=16 来自一次 10 档私有探针（(4096,4096)/(1024,65536)，int16+int32）：
    BLOCK 2048→4096 提升 8%~15%（b1024 反而掉到 0.17x/0.39x），
    CHUNK 1→8 提升 ~80%（归约是主要开销），CHUNK 16 在 int32 上再快 ~9%、int16 基本持平；
    num_warps 2/4/8 与 grid 2x/4x 无可测差异，grid 1x 明显变差。
    """
    if dtype == torch.int64:
        return gcd_kernel_i64, 256, 4, 4
    if dtype in (torch.int8, torch.int16, torch.int32):
        return gcd_kernel_i32, 4096, 4, 16
    raise TypeError(f"unsupported dtype for gcd: {dtype}")


def _dtype_min(dtype):
    if dtype == torch.int64:
        return -(1 << 63)
    return torch.iinfo(dtype).min


def _materialize_inputs(self, other):
    promoted_dtype = torch.promote_types(self.dtype, other.dtype)
    lhs = self if self.dtype == promoted_dtype else self.to(promoted_dtype)
    rhs = other if other.dtype == promoted_dtype else other.to(promoted_dtype)
    lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    return lhs.contiguous(), rhs.contiguous(), promoted_dtype


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out

    kernel, block, num_warps, chunk = _kernel_meta(out.dtype)
    grid = min((numel + block - 1) // block, NUM_SIPS * 2)
    with torch_device_fn.device(out.device):
        kernel[(grid,)](
            lhs,
            rhs,
            out,
            numel,
            DTYPE_MIN=_dtype_min(out.dtype),
            BLOCK=block,
            CHUNK=chunk,
            num_warps=num_warps,
        )
    return out


def gcd(self, other, *, out=None):
    logger.debug("GEMS_ENFLAME GCD")
    lhs, rhs, promoted_dtype = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), result.reshape(-1))
    result = result.view(lhs.shape)
    if out is None:
        return result
    out.copy_(result)
    return out


def gcd_out(self, other, *, out=None):
    logger.debug("GEMS_ENFLAME GCD_OUT")
    if out is None:
        return gcd(self, other)
    return gcd(self, other, out=out)
