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

"""gcu400 linalg_matmul: the common implementation with the backend GEMMs.

The common ``flag_gems.ops.linalg_matmul`` imports ``flag_gems.ops.mm`` and
``flag_gems.ops.bmm`` directly.  Those generic kernels do not accept the
gcu400 tuned-config keys (``SPLIT_K`` / ``BLOCK_*``), so every call dies with
``KeyError: Keyword argument SPLIT_K was specified but unrecognised``.  The
backend already ships matching ``mm`` / ``bmm`` kernels; this module wires them
into the same shape-folding and autograd logic, and adds the zero-sized fast
exit that a 2-D matmul with an empty M or N needs (a zero grid is not a legal
launch).

The batched arm composes the backend ``bmm`` op, whose hybrid dispatch routes
saturated problems (``B*M*N*K >= 2^29``) to the native primo++ kernel captured
at import time (bit-exact, near-zero dispatch overhead) and keeps the Triton
``bmm_kernel`` below the line; launching the Triton kernel for every size left
the large core cases at 0.51-0.84x of torch (see ``bmm.py`` for the sweep).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.bmm import bmm
from flag_gems.runtime.backend._enflame.gcu400.ops.mm import mm

logger = logging.getLogger(__name__)


@triton.jit
def _tiny_mm_kernel(
    A,
    B,
    C,
    N,
    K,
    sa0,
    sa1,
    sb0,
    sb1,
    sc0,
    sc1,
    BLOCK_K: tl.constexpr,
):
    """One program per output element, plain multiply-accumulate.

    The fp32 ``tl.dot`` path fails to link on this backend for 1-row/1-column
    products (``ld.lld: relocation R_DTU_ADDR16_LO_ICALL ... recompile with
    -fPIC`` from the int8 conversion helpers), so those shapes avoid tl.dot.
    """
    pid = tl.program_id(0)
    m = pid // N
    n = pid % N
    kk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        idx = k0 + kk
        mask = idx < K
        a = tl.load(A + m * sa0 + idx * sa1, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + idx * sb0 + n * sb1, mask=mask, other=0.0).to(tl.float32)
        acc += a * b
    tl.store(C + m * sc0 + n * sc1, tl.sum(acc, axis=0).to(C.dtype.element_ty))


def _tiny_mm(a, b):
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    BLOCK_K = max(16, min(triton.next_power_of_2(K), 1024))
    with torch_device_fn.device(a.device):
        _tiny_mm_kernel[(M * N,)](
            a,
            b,
            out,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            BLOCK_K=BLOCK_K,
        )
    return out


def _linalg_bmm(a, b):
    """复用本 backend 的 bmm 算子（组合复用，不改动 bmm 通用实现）。

    bmm 内部已是混合分派：饱和段（B*M*N*K >= 2^29）委托 import 期捕获的
    native primo++ kernel（capture + call_boxed，bit 级一致、近零派发开销），
    线下段走 bmm_kernel 的确定性 tile 分档——与本模块此前直接启动的是同一个
    kernel、同一套小形状档位。此前本函数对一切尺寸直接启动 Triton kernel，
    绕过了该分派，令 core bench 的 12 个大 case 只有 0.51-0.84x：Triton GEMM
    吞吐结构性封顶在 145-165 TFLOPS，native pipeline 到 276-297 TFLOPS
    （10 配置扫描确认无调参空间，bmm job j20261111073819_8d5ecfcd，
    mm/baddbmm/router_gemm 同款分派）。
    """
    return bmm(a, b)


class LinalgMatmulFunction(torch.autograd.Function):
    """Autograd for linalg_matmul.

    Forward -> shape manipulation plus the Triton mm/bmm kernels.
    Backward -> recompute the product in native PyTorch under
    torch.enable_grad, then take gradients with torch.autograd.grad,
    so 1-D folding and batch-dim broadcasting follow native matmul
    gradient semantics.
    """

    @staticmethod
    def forward(ctx, input, other):
        # matmul is only defined for floating point dtypes; reject integral
        # and complex inputs with errors matching the ATen CUDA backend
        # instead of letting them fail inside the Triton kernel.
        if not input.is_floating_point():
            raise NotImplementedError(
                f"linalg_matmul not implemented for '{input.dtype}' on this device"
            )
        if not other.is_floating_point():
            raise NotImplementedError(
                f"linalg_matmul not implemented for '{other.dtype}' on this device"
            )
        if input.dtype != other.dtype:
            raise RuntimeError(
                f"expected mat1 and mat2 to have the same dtype, but got: "
                f"{input.dtype} != {other.dtype}"
            )

        ctx.save_for_backward(input, other)
        return _linalg_matmul_forward(input, other)

    @staticmethod
    def backward(ctx, grad_output):
        input, other = ctx.saved_tensors
        orig_dtype = input.dtype
        # run the backward in fp32 for half-precision inputs to preserve
        # gradient precision
        compute_dtype = (
            torch.float32
            if orig_dtype in (torch.float16, torch.bfloat16)
            else orig_dtype
        )
        # the enable_grad block builds a graph for the recomputed forward, so
        # higher-order gradients flow through this backward as well
        with torch.enable_grad():
            input = input.detach().to(compute_dtype).requires_grad_(True)
            other = other.detach().to(compute_dtype).requires_grad_(True)
            output = torch.matmul(input, other)
            grad_input, grad_other = torch.autograd.grad(
                output,
                (input, other),
                grad_output.to(compute_dtype),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )
        return (
            None if grad_input is None else grad_input.to(orig_dtype),
            None if grad_other is None else grad_other.to(orig_dtype),
        )


def _empty_matmul_out(shape, dtype, device, fold_first, fold_last):
    out = torch.empty(shape, dtype=dtype, device=device)
    if fold_first:
        out = out.squeeze(-2)
    if fold_last:
        out = out.squeeze(-1)
    return out


def _linalg_matmul_forward(input, other):
    logger.debug("GEMS_ENFLAME LINALG_MATMUL")

    # Fold 1D inputs into 2D per torch.matmul semantics:
    # (K,) @ (K, N) -> (N,) and (M, K) @ (K,) -> (M,), (K,) @ (K,) -> scalar
    fold_first = input.dim() == 1
    fold_last = other.dim() == 1
    if fold_first:
        input = input.unsqueeze(0)
    if fold_last:
        other = other.unsqueeze(1)

    # Handle 2D case: (M, K) @ (K, N) -> (M, N)
    if input.dim() == 2 and other.dim() == 2:
        M, K = input.shape
        N = other.shape[1]
        if M == 0 or N == 0 or K == 0:
            # No work to do, and a Triton launch with a zero grid is invalid.
            return _empty_matmul_out(
                (M, N), input.dtype, input.device, fold_first, fold_last
            )
        if M == 1 or N == 1:
            out = _tiny_mm(input, other)
        else:
            out = mm(input, other)
        if fold_first and fold_last:
            return out.reshape(())
        if fold_first:
            return out[0]
        if fold_last:
            return out[:, 0]
        return out

    # Handle batched cases: broadcast the batch dims, then run a single bmm
    # over the flattened batch. expand produces stride-0 views which the
    # bmm kernel handles natively.
    assert input.dim() >= 2 and other.dim() >= 2, "incompatible dimensions"
    bshape = torch.broadcast_shapes(input.shape[:-2], other.shape[:-2])
    M, K = input.shape[-2], input.shape[-1]
    N = other.shape[-1]
    assert K == other.shape[-2], "incompatible dimensions"
    a = input.expand(bshape + (M, K))
    b = other.expand(bshape + (K, N))
    nbatch = 1
    for s in bshape:
        nbatch *= s
    if nbatch == 0 or M == 0 or N == 0 or K == 0:
        # empty batch / empty matrix: no kernel launch (grid of 0 is invalid),
        # just an empty output with the folded 1-D dims restored
        return _empty_matmul_out(
            bshape + (M, N), input.dtype, input.device, fold_first, fold_last
        )
    out = _linalg_bmm(a.reshape(nbatch, M, K), b.reshape(nbatch, K, N))
    out = out.reshape(bshape + (M, N))
    if fold_first:
        out = out.squeeze(-2)
    if fold_last:
        out = out.squeeze(-1)
    return out


def linalg_matmul(input, other):
    """Matrix product of two tensors.

    The behavior depends on the dimensionality of the inputs:
    - If both inputs are 2D, performs regular matrix multiplication: (M, K) @ (K, N) -> (M, N)
    - If both inputs are 3D (batched), performs batched matrix multiplication:
      (B, M, K) @ (B, K, N) -> (B, M, N)
    - Batch dims are broadcast against each other, and 1D inputs are folded
      into the product per torch.matmul semantics.

    This is an alias for torch.linalg.matmul.
    """
    logger.debug("GEMS_ENFLAME LINALG_MATMUL")
    return LinalgMatmulFunction.apply(input, other)
