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

from flag_gems import runtime
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native-kernel delegation (capture + call_boxed), same pattern as the local
# _linalg_eigvals / linalg_multi_dot / linalg_cond vendor ops.
#
# Measured on L600 (bench kernel mode, do_bench median w1000/r100, cold L2):
# the Triton kernel beats the native vendor GEMM only while the problem is
# launch/overhead bound; once the native primo++ pipeline saturates it is
# ~2x faster than any swept Triton config ((128,256,256) is the tile sweet
# spot, matching the matrix_power job's 50+ combo sweep):
#   (2,384^3)   Triton 1.08-1.19x   (2,512^3)  Triton 1.12-1.17x
#   (2,640^3)   Triton 0.96x        (2,768^3)  Triton 0.93x
#   (4,512^3)   Triton 0.64x        (16,384^3) Triton 0.57x
#   (16,1024^3) Triton 0.51x        (16,4096^3) Triton 0.55x
# The kernels below are captured at import time - before any use_gems()
# registration replaces the PrivateUse1 entry - so call_boxed runs the
# original device kernel bit-identically with zero dispatch overhead
# (verified 1.000-1.006x vs plain torch.bmm on the shapes above).
# Dispatch line: total work B*M*N*K >= 2^29 routes to the native kernel;
# both sides of the line have measured anchor points (see table above).
# ---------------------------------------------------------------------------
_CAN_CAPTURE_NATIVE_KERNEL = hasattr(torch.library, "get_kernel")
_NATIVE_BMM = None
_NATIVE_BMM_OUT = None
if _CAN_CAPTURE_NATIVE_KERNEL:
    try:
        _NATIVE_BMM = torch.library.get_kernel(
            torch.ops.aten.bmm.default, runtime.device.dispatch_key
        )
        _NATIVE_BMM_OUT = torch.library.get_kernel(
            torch.ops.aten.bmm.out, runtime.device.dispatch_key
        )
    except Exception:  # noqa: B902
        _NATIVE_BMM = None
        _NATIVE_BMM_OUT = None

_DEVICE_KEYSET = torch._C.DispatchKeySet(
    getattr(torch._C.DispatchKey, runtime.device.dispatch_key)
)
_NATIVE_WORK_THRESHOLD = 1 << 29


def _should_delegate_to_native(A, B):
    if _NATIVE_BMM is None:
        return False
    if A.device.type != runtime.device.name:
        return False
    Batch, M, K = A.shape
    N = B.shape[2]
    return Batch * M * N * K >= _NATIVE_WORK_THRESHOLD


# Hardware DSM (distributed shared memory) budget of one GCU400 cluster. The
# staged A/B tiles of every software-pipeline stage live there, so a tile table
# that ignores this limit fails to compile ("out of resource: dsm memory").
_DSM_LIMIT_BYTES = 512 * 1024


def _pow2_ceil(value):
    return 1 << max(0, int(value) - 1).bit_length()


def _clamp_tile(dim, tile, floor=64):
    # Never request a tile far larger than the problem dimension. Oversized
    # tiles both waste DSM and overflow it: (B, 64, 1, 4096) used to request a
    # 256-wide N tile for a single output column and failed to compile.
    #
    # The floor of 64 is a correctness guard, not just padding: on GCU400 the
    # unmasked (EVEN) store/download path silently computes wrong values for
    # BLOCK_M in {16, 32} combined with BLOCK_N >= 256, e.g. (2,32,512,4096)
    # with (32,256,256) returns relerr ~2.7 while (32,128,256) and
    # (64,256,256) are exact. Keeping BLOCK_M >= 64 avoids that region.
    if dim >= tile:
        return tile
    return max(floor, min(tile, _pow2_ceil(dim)))


def _fit_num_stages(block_m, block_n, block_k, elem_bytes, num_stages):
    # Shrink the software pipeline until the staged tiles fit the DSM budget.
    while num_stages > 1:
        staged = (block_m * block_k + block_k * block_n) * elem_bytes * num_stages
        if staged <= _DSM_LIMIT_BYTES:
            break
        num_stages -= 1
    return num_stages


def _select_gemm_config(Batch, M, N, K, dtype):
    # Single source of truth for the GEMM tile table, shared by bmm / bmm_out /
    # baddbmm / baddbmm_out / baddbmm_ (all of them compute the same batched
    # GEMM core; baddbmm only adds a stride-based bias epilogue).
    #
    # Tiles were swept per shape tier on GCU400 (4-1, 2 clusters x 12 SIPs):
    #   * (2,384^3)  fp16/bf16: (128,128,512,w4,ns2) 13.4us vs (128,128,256) 14.7us
    #     vs the previous (256,128,128) 16.9us and (64,128,128,w1) 25.3us.
    #     A 512-deep K block keeps the inner loop at one masked iteration.
    #   * (2,384^3)  fp32: (128,128,512,w2,ns2) is the same shape as the fp16 win.
    #   * saturated shapes (>= 4096^2, batch >= 8) keep the wide fp16 tile.
    num_stages = 3
    if K < 128:
        block_m, block_n, block_k, num_warps, num_stages = 64, 64, 32, 4, 2
    elif M <= 512 and N <= 512 and K <= 512:
        block_m, block_n, block_k, num_warps, num_stages = 128, 128, 512, 4, 2
    elif M >= 4096 and N >= 4096 and Batch >= 8:
        if dtype == torch.float32:
            block_m, block_n, block_k, num_warps = 256, 128, 128, 4
        else:
            block_m, block_n, block_k, num_warps = 128, 256, 128, 2
    elif dtype == torch.float32:
        block_m, block_n, block_k, num_warps = 128, 256, 128, 2
    else:
        block_m, block_n, block_k, num_warps = 128, 256, 256, 4

    block_m = _clamp_tile(M, block_m)
    block_n = _clamp_tile(N, block_n)
    block_k = _clamp_tile(K, block_k, floor=16)
    if block_n > 256 and block_m < 128:
        # Keep the wide-N tiles on tall enough blocks; see _clamp_tile notes.
        block_n = 256
    elem_bytes = 4 if dtype in (torch.float32, torch.float64) else 2
    num_stages = _fit_num_stages(block_m, block_n, block_k, elem_bytes, num_stages)

    # Exact decision: only when all three dimensions are divisible by the actually selected tiles are load/store
    # free of out-of-bounds access, in which case an unmasked fast path is compiled; otherwise use the masked generic
    # path.
    even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
    return block_m, block_n, block_k, num_warps, num_stages, even


# Conservative last-resort tile used when the primary config exceeds the DSM
# budget at compile time (keeps the operator usable instead of raising).
_SAFE_FALLBACK_CONFIG = (64, 64, 64, 4, 2)


def _select_bmm_config(Batch, M, N, K, dtype):
    return _select_gemm_config(Batch, M, N, K, dtype)


@libentry()
@triton.jit
def bmm_kernel(
    A,
    B,
    C,
    Batch,
    M,
    N,
    K,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + pid_b * stride_ab
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # EVEN is a compile-time constant, so the compiler statically trims it to a single loop,
    # avoiding a runtime branch interrupting the tl.dot pipeline (a runtime branch degrades dot to scalar compute).
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        C
        + pid_b * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    c = acc.to(C.dtype.element_ty)
    if EVEN:
        tl.store(c_ptrs, c)
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _launch_bmm(A, B, out, Batch, M, N, K):
    # Centralize config dispatch and kernel launch, shared by bmm / bmm_out.
    block_m, block_n, block_k, num_warps, num_stages, even = _select_bmm_config(
        Batch, M, N, K, A.dtype
    )
    _run_bmm_kernel(
        A,
        B,
        out,
        Batch,
        M,
        N,
        K,
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        even,
    )


def _run_bmm_kernel(
    A, B, out, Batch, M, N, K, block_m, block_n, block_k, num_warps, num_stages, even
):
    grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    launch = dict(
        dot_out_dtype=tl.float32,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=8,
        EVEN=even,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    args = (
        A,
        B,
        out,
        Batch,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    try:
        bmm_kernel[grid](*args, **launch)
    except triton.runtime.errors.OutOfResources:
        # The selected tiles do not fit the hardware DSM budget for this shape;
        # retry once with the conservative tile instead of failing the operator.
        block_m, block_n, block_k, num_warps, num_stages = _SAFE_FALLBACK_CONFIG
        block_m = _clamp_tile(M, block_m)
        block_n = _clamp_tile(N, block_n)
        block_k = _clamp_tile(K, block_k, floor=16)
        even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
        grid = (Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
        bmm_kernel[grid](
            *args,
            dot_out_dtype=tl.float32,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            EVEN=even,
            num_warps=num_warps,
            num_stages=num_stages,
        )


def _as_gemm_friendly(A, B):
    # The kernel indexes A as [m, k] and B as [k, n] using explicit strides, so
    # any layout with a unit stride along the *inner* dot axis is fed directly.
    # Forcing .contiguous() on a transposed B used to materialize a copy through
    # an extra kernel: (2,512^3) measured 24.1us copy+kernel vs 19.5us strided
    # (native 25.1us). Only layouts with no unit stride on either axis (e.g. a
    # sliced/broadcast view) still get a copy.
    if A.stride(2) != 1 and not A.is_contiguous():
        A = A.contiguous()
    if B.stride(2) != 1 and B.stride(1) != 1 and not B.is_contiguous():
        B = B.contiguous()
    return A, B


def bmm(A, B):
    logger.debug("GEMS_ENFLAME BMM")
    assert A.shape[0] == B.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    # Saturation range (large workload) delegates to the native kernel: bit-identical, zero dispatch overhead;
    # Passed through unchanged (native supports arbitrary stride, so non-contiguous inputs need no contiguous copy).
    if _should_delegate_to_native(A, B):
        return _NATIVE_BMM.call_boxed(_DEVICE_KEYSET, A, B)
    Batch, M, K = A.shape
    _, _, N = B.shape
    A, B = _as_gemm_friendly(A, B)
    out = torch.empty((Batch, M, N), dtype=A.dtype, device=A.device)
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out


def bmm_out(A, B, out):
    logger.debug("GEMS_ENFLAME BMM_OUT")
    assert A.shape[0] == B.shape[0] == out.shape[0], "Batch dim mismatch"
    assert A.shape[2] == B.shape[1], "K dim mismatch"
    if _should_delegate_to_native(A, B) and _NATIVE_BMM_OUT is not None:
        return _NATIVE_BMM_OUT.call_boxed(_DEVICE_KEYSET, A, B, out=out)
    Batch, M, K = A.shape
    _, _, N = B.shape
    A, B = _as_gemm_friendly(A, B)
    _launch_bmm(A, B, out, Batch, M, N, K)
    return out
