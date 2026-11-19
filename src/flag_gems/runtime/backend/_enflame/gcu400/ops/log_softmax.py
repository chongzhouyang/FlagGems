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
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N / conf.num_warps > 64 * 512:
        return False
    if BLOCK_M * BLOCK_N < 64 * 64:
        return False
    if conf.num_stages > 1:
        return False
    if conf.num_warps == 1:
        return False
    return True


@libentry()
@libtuner(
    configs=list(filter(keep, runtime.get_tuned_config("log_softmax"))),
    key=["M", "N"],
)
@triton.jit
def log_softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr = 16,
    BLOCK_N: tl.constexpr = 2048,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    for tile_k_id in tl.range(pid_k, K, 48):
        # TODO(chenfeiyu): consider float64 add add a utility function to get accumulator type
        m = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
        z = tl.full([BLOCK_M, BLOCK_N], value=0.0, dtype=tl.float32)
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + tile_k_id
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            m_new = tl.maximum(inp, m)
            all_neg_inf = m_new == float("-inf")
            z = tl.where(all_neg_inf, z, z * tl.exp(m - m_new) + tl.exp(inp - m_new))
            m = m_new

        m_reduced = tl.max(m, 1)
        z = tl.sum(z * tl.exp(m - m_reduced[:, None]), 1)
        m = m_reduced
        mid = m[:, None] + tl.log(z[:, None])
        for start_n in range(0, N, BLOCK_N):
            n_offset = start_n + tl.arange(0, BLOCK_N)
            offset = m_offset[:, None] * N * K + n_offset[None, :] * K + tile_k_id
            mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf")).to(tl.float32)
            o = inp - mid
            tl.store(output_ptr + offset, o, mask=mask)


@libentry()
@triton.jit
def log_softmax_fused_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """N <= BLOCK_N 的单遍 kernel：整行驻留寄存器，1 次读 + 1 次写。

    通用 kernel 需要读两遍输入（一遍求 max/sumexp，一遍算输出），
    在 N 能被一个 tile 覆盖时单遍即可，global 访问从 3 次降到 2 次。
    """
    pid = tl.program_id(0)
    m_offset = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K
    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    inp = tl.load(input_ptr + offsets, mask=mask, other=float("-inf")).to(tl.float32)
    row_max = tl.max(inp, 1)
    # When an entire row is outside the mask, row_max is -inf and exp(-inf - (-inf)) produces NaN;
    # replace such a row with 0 (there are no valid elements in it, and its output is masked out anyway).
    row_max = tl.where(row_max == float("-inf"), 0.0, row_max)
    row_sum = tl.sum(tl.exp(inp - row_max[:, None]), 1)
    out = inp - (row_max + tl.log(row_sum))[:, None]
    tl.store(output_ptr + offsets, out, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Keep only one scalar accumulator per row, avoiding a whole BLOCK_M x BLOCK_N resident register block.
    scale = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_grad_ptrs = out_grad_ptr + offsets
        # Masked regions must be filled with 0, or undefined values pollute the whole row through the row sum.
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        scale += tl.sum(out_grad, 1)

    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
        out_ptrs = out_ptr + offsets
        out = tl.load(out_ptrs, mask=mask, other=0.0).to(tl.float32)
        out_grad_ptrs = out_grad_ptr + offsets
        out_grad = tl.load(out_grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        in_grad = out_grad - tl.exp(out) * scale[:, None]
        in_grad_ptrs = in_grad_ptr + offsets
        tl.store(in_grad_ptrs, in_grad, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_fused_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-pass backward for rows that fit in one tile.

    The generic kernel reads grad_output twice (once for the row sum, once for
    the output).  When N <= BLOCK_N the whole row can be kept in registers, so
    one pass is enough: 3 global accesses per element instead of 4.
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    offsets = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
    mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    out_grad = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.sum(out_grad, 1)
    in_grad = out_grad - tl.exp(out) * scale[:, None]
    tl.store(in_grad_ptr + offsets, in_grad, mask=mask)


@libentry()
@triton.jit
def log_softmax_backward_n1_kernel(
    out_ptr,
    out_grad_ptr,
    in_grad_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    """归约宽度 N == 1 的逐元素退化路径。

    每行只有一个元素，softmax 分母恒为 1，因此
        in_grad = out_grad - exp(out) * sum(out_grad) = out_grad * (1 - exp(out))
    逐元素即可精确算出。通用 kernel 在这种形状上每行都要走完整的
    归约流程（grid 第二维退化为 1），实测比 aten 慢 ~10x。
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    out_grad = tl.load(out_grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    in_grad = out_grad * (1.0 - tl.exp(out))
    tl.store(in_grad_ptr + offsets, in_grad, mask=mask)


def log_softmax_out(self, dim, half_to_float=False, *, out):
    """K=1 时复用 gcu400 online-softmax kernel 直写 out；K>1 保留公共 non_inner。

    公共 pointwise 的 2D tiled kernel 在 K=1 上比 gcu400 kernel 慢 2.3~4.5x
    （4096² 598.8µs vs 173.3µs）；但 K>1（非 inner dim 归约）时公共的非 inner
    kernel 反过来快 7.6~14x，因此按 K 分档而不是一刀切。
    """
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_OUT")
    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    dtype = torch.float32 if half_to_float else self.dtype
    if tuple(out.shape) != tuple(inp.shape):
        out.resize_(inp.shape)
    if out.dtype != dtype:
        raise RuntimeError(
            f"_log_softmax.out: expected out dtype {dtype}, got {out.dtype}"
        )
    if N == 1:
        out.zero_()
        return out
    K = inp.numel() // M // N
    if K > 1:
        from flag_gems.ops.log_softmax import log_softmax_out as _public_out

        return _public_out(self, dim, half_to_float, out=out)

    # When a row can reside entirely in registers, use a single-pass kernel: 2 global accesses instead of 3.
    fused_launch = select_forward_fused_launch(N, inp.dtype)
    if fused_launch is not None:
        block_m, num_warps = fused_launch
        kernel_out = out if out.is_contiguous() else torch.empty_like(inp, dtype=dtype)
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(inp.device):
            log_softmax_fused_kernel[grid](
                kernel_out,
                inp,
                M,
                N,
                K,
                BLOCK_M=block_m,
                BLOCK_N=triton.next_power_of_2(N),
                num_warps=num_warps,
            )
        return _copy_back_if_needed(out, kernel_out)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), min(48, K))
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](out, inp, M, N, K)
    return out


def log_softmax(self, dim, half_to_float=False):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX")

    assert dim >= -self.ndim and dim < self.ndim, "Invalid dim"
    dim = dim % self.ndim
    M = 1
    N = self.shape[dim]
    for i in range(dim):
        M *= self.shape[i]
    inp = self.contiguous()
    if half_to_float:
        dtype = torch.float32
    else:
        dtype = self.dtype

    if N == 1:
        return torch.zeros_like(inp, dtype=dtype)

    K = inp.numel() // M // N
    if K > 1:
        # The tuned kernel reduces one k-plane per program; for K > 1 the
        # public non-inner kernel is both faster and accurate in 16-bit
        # (the tuned path lost ~7% of the row sum in fp16/bf16 at N=40999).
        from flag_gems.ops.log_softmax import log_softmax as _public

        return _public(self, dim, half_to_float)
    out = torch.empty_like(inp, dtype=dtype)

    # Same as log_softmax_out: when a row can reside entirely in registers, use a single-pass kernel, saving the second
    # input read.
    fused_launch = select_forward_fused_launch(N, inp.dtype)
    if fused_launch is not None:
        block_m, num_warps = fused_launch
        grid = (triton.cdiv(M, block_m),)
        with torch_device_fn.device(inp.device):
            log_softmax_fused_kernel[grid](
                out,
                inp,
                M,
                N,
                K,
                BLOCK_M=block_m,
                BLOCK_N=triton.next_power_of_2(N),
                num_warps=num_warps,
            )
        return out

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        min(48, K),
    )
    with torch_device_fn.device(inp.device):
        log_softmax_kernel[grid](
            out,
            inp,
            M,
            N,
            K,
        )
    return out


# === LAUNCH_CONFIG_BEGIN ===
def select_fused_launch(N):
    """Choose rows per program, warp count and column tile for the single-pass kernel by reduction width."""
    if N <= 256:
        return 32, 2, triton.next_power_of_2(N)
    if N <= 1024:
        return 64, 4, triton.next_power_of_2(N)
    return 8, 4, triton.next_power_of_2(N)


def select_twopass_launch(N, dtype):
    """按归约宽度与 dtype 选择两遍 kernel 的行/列 tile 与 warp 数。"""
    if dtype == torch.float32:
        return 8, min(triton.next_power_of_2(N), 8192), 4
    # A 16-row column tile of fp16/bf16 reaches 131072 elements at N > 8192 and the
    # compiler spills straight into 320KB of local memory (localTotalSize=327680),
    # measuring 0.684x at N=65536. Capping the column tile at 4096 and raising
    # num_warps to 8 gives 0.751x; N <= 8192 keeps the original setting so the
    # already validated small widths are unaffected.
    cap = 4096 if N > 8192 else 8192
    return 16, min(triton.next_power_of_2(N), cap), 8


def select_forward_fused_launch(N, dtype):
    """单遍前向 kernel 的行 tile 与 warp 数；返回 None 表示走两遍 kernel。

    单遍 kernel 的整行要驻留寄存器，因此按“每 program 元素数预算”反推行数，
    预算按 dtype 的字节宽度定（fp32 只有 fp16/bf16 的一半）。超出预算后
    tile 会溢出到局部内存，反而不如两遍 kernel —— 实测 fp32 在 N > 512、
    fp16/bf16 在 N > 4096 时两遍更好，这两档返回 None。
    """
    block_n = triton.next_power_of_2(N)
    if dtype == torch.float32:
        if block_n > 512:
            return None
        element_budget = 8192
    else:
        if block_n > 4096:
            return None
        element_budget = 16384
    block_m = max(1, min(128, element_budget // block_n))
    return block_m, 8


# === LAUNCH_CONFIG_END ===


def _copy_back_if_needed(out, kernel_out):
    """Copy the result from a contiguous temporary tensor back into a possibly non-contiguous out view."""
    if kernel_out is not out:
        out.copy_(kernel_out)
    return out


def log_softmax_backward_out(grad_output, output, dim, input_dtype, *, out):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_BACKWARD_OUT")
    assert dim >= -output.ndim and dim < output.ndim, "Invalid dim"
    dim = dim % output.ndim
    M = 1
    N = output.shape[dim]
    for i in range(dim):
        M *= output.shape[i]

    if tuple(out.shape) != tuple(output.shape):
        out.resize_(output.shape)
    if out.dtype != input_dtype:
        raise RuntimeError(
            f"_log_softmax_backward_data.out: expected out dtype {input_dtype}, got {out.dtype}"
        )
    # An empty tensor has no reduction elements; return directly to avoid division by zero in K = numel // M // N.
    if output.numel() == 0:
        return out

    # The kernel computes offsets assuming a contiguous memory layout: non-contiguous inputs are converted to contiguous
    # copies,
    # and non-contiguous out is written to a contiguous temporary then copied back; Triton remains the only compute
    # path.
    grad_output = grad_output.contiguous()
    output = output.contiguous()
    kernel_out = (
        out if out.is_contiguous() else torch.empty_like(output, dtype=input_dtype)
    )
    K = output.numel() // M // N

    # N == 1: one element per row, result is out_grad * (1 - exp(out)), which can be computed exactly elementwise.
    if N == 1:
        n_elements = grad_output.numel()
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        with torch_device_fn.device(out.device):
            log_softmax_backward_n1_kernel[grid](
                output, grad_output, kernel_out, n_elements, BLOCK=block, num_warps=4
            )
        return _copy_back_if_needed(out, kernel_out)

    # Single-pass fused path for rows that fit one tile (N <= 4096).
    # Saves one full grad_output read: 4 global accesses -> 3.
    # Tile size is chosen per N: small rows want more rows per program and
    # fewer warps, wide rows want fewer rows to keep the tile in registers.
    if K == 1 and N <= 4096:
        block_m, num_warps, block_n = select_fused_launch(N)
        grid = (triton.cdiv(M, block_m), K)
        with torch_device_fn.device(out.device):
            log_softmax_backward_fused_kernel[grid](
                output,
                grad_output,
                kernel_out,
                M,
                N,
                K,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_warps=num_warps,
            )
        return _copy_back_if_needed(out, kernel_out)

    # Two-pass path for wide rows (or K > 1).  Autotuning was unstable here:
    # the first two benchmark rounds picked a 530us config while later rounds
    # picked 383us.  A fixed tile is both deterministic and faster to compile.
    block_m, block_n, num_warps = select_twopass_launch(N, output.dtype)
    grid = (triton.cdiv(M, block_m), K)
    with torch_device_fn.device(out.device):
        log_softmax_backward_kernel[grid](
            output,
            grad_output,
            kernel_out,
            M,
            N,
            K,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    return _copy_back_if_needed(out, kernel_out)


def log_softmax_backward(grad_output, output, dim, input_dtype):
    logger.debug("GEMS_ENFLAME LOG_SOFTMAX_VJP")
    in_grad = torch.empty_like(output, dtype=input_dtype)
    return log_softmax_backward_out(grad_output, output, dim, input_dtype, out=in_grad)
