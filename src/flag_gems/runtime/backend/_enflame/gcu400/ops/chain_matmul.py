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

"""chain_matmul in Triton.

Computes the product of a chain of 2-D matrices using an optimal (minimum-flop)
parenthesization, mirroring torch.linalg.multi_dot / aten::chain_matmul
semantics. The host wrapper only runs the matrix-chain DP (shape metadata)
and allocates nothing beyond what the callee allocates; every GEMM node is
delegated to this backend's mm operator (single source for the native
delegation line and the tuned Triton GEMM segment). Each DP node rounds back
to the input dtype inside mm, so the result matches a native-dtype reference
rather than an internally-widened one.

The ``out`` variant writes the final GEMM straight into the caller's buffer
through ``mm_out`` (native out= delegation where eligible, else a
constexpr-strided Triton store), so the chain result is never copied a second
time. Only the autograd path, which needs its own result tensor, pays an
extra copy into ``out``.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.mm import mm, mm_out

logger = logging.getLogger(__name__)


@triton.jit
def _copy(
    A,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    AM: tl.constexpr,
    AN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
):
    x = tl.program_id(0) * 256 + tl.arange(0, 256)
    m = x // N
    n = x % N
    v = tl.load(A + m * AM + n * AN, x < M * N, 0)
    tl.store(C + m * CM + n * CN, v, x < M * N)


def _copy2d(a, c):
    m, n = a.shape
    if m * n:
        _copy[(triton.cdiv(m * n, 256),)](a, c, m, n, *a.stride(), *c.stride())
    return c


def _split(p):
    n = len(p) - 1
    cost = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]
    for length in range(1, n):
        for i in range(n - length):
            j = i + length
            best = None
            for k in range(i, j):
                v = cost[i][k] + cost[k + 1][j] + p[i] * p[k + 1] * p[j + 1]
                if best is None or v < best:
                    best = v
                    s[i][j] = k
            cost[i][j] = best
    return s


@lru_cache(maxsize=128)
def _schedule(p):
    # Only immutable integer metadata is retained, never tensors or values.
    s = _split(p)
    order = []

    def build(i, j):
        if i != j:
            k = s[i][j]
            build(i, k)
            build(k + 1, j)
            order.append((i, j, k))

    build(0, len(p) - 2)
    return tuple(order)


def _tree(mats):
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    order = _schedule(p)
    t = {(i, i): a for i, a in enumerate(mats)}
    for i, j, k in order:
        t[i, j] = mm(t[i, k], t[k + 1, j])
    return t[0, len(mats) - 1], t, order


def _forward(mats, out=None):
    if len(mats) == 1:
        return _copy2d(mats[0], out if out is not None else torch.empty_like(mats[0]))
    if len(mats) == 2:
        if out is not None:
            return mm_out(mats[0], mats[1], out=out)
        return mm(mats[0], mats[1])
    p = tuple(a.shape[0] for a in mats) + (mats[-1].shape[1],)
    order = _schedule(p)
    t = {(i, i): a for i, a in enumerate(mats)}
    last = len(order) - 1
    for step, (i, j, k) in enumerate(order):
        a, b = t[i, k], t[k + 1, j]
        if out is not None and step == last:
            t[i, j] = mm_out(a, b, out=out)
        else:
            t[i, j] = mm(a, b)
    return t[0, len(mats) - 1]


class _Chain(torch.autograd.Function):
    @staticmethod
    def forward(*mats):
        return _forward(mats)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad):
        mats = ctx.saved_tensors
        n = len(mats)
        if n == 1:
            return (grad,)
        _, t, order = _tree(mats)
        grads = {(0, n - 1): grad}
        for i, j, k in reversed(order):
            g = grads.pop((i, j))
            grads[i, k] = mm(g, t[k + 1, j].t())
            grads[k + 1, j] = mm(t[i, k].t(), g)
        return tuple(grads[i, i] for i in range(n))


def run(matrices, *args, out=None):
    logger.debug("GEMS_ENFLAME CHAIN_MATMUL")
    if isinstance(matrices, torch.Tensor):
        mats = [matrices, *args]
    elif isinstance(matrices, (list, tuple)):
        if args:
            raise TypeError("chain_matmul(): unexpected extra arguments")
        mats = list(matrices)
    else:
        raise TypeError("chain_matmul(): expected a list of tensors")
    if not mats:
        raise RuntimeError("chain_matmul(): Expected one or more matrices")
    if any(not isinstance(a, torch.Tensor) for a in mats):
        raise TypeError("expected tensors")
    dtype, device = mats[0].dtype, mats[0].device
    for i, a in enumerate(mats):
        if a.ndim != 2:
            raise RuntimeError(f"Tensor dimension is {a.ndim}, expected 2 instead.")
        if a.dtype != dtype or a.device != device:
            raise RuntimeError(
                "multi_dot(): all tensors must have the same dtype and device"
            )
        if not dtype.is_floating_point:
            raise RuntimeError("chain_matmul(): unsupported dtype")
        if i and mats[i - 1].shape[1] != a.shape[0]:
            raise RuntimeError("multi_dot(): tensors cannot be multiplied")
    if out is not None:
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("out must be a Tensor")
        if out.dtype != dtype or out.device != device:
            raise RuntimeError("multi_dot(): out dtype or device mismatch")
        if tuple(out.shape) != (mats[0].shape[0], mats[-1].shape[1]):
            raise RuntimeError("multi_dot(): out shape mismatch")
    result = (
        _Chain.apply(*mats)
        if torch.is_grad_enabled() and any(a.requires_grad for a in mats)
        else _forward(mats, out)
    )
    if out is None:
        return result
    if result is not out:
        _copy2d(result, out)
    return out


def chain_matmul_out(matrices, *args, out):
    """aten::chain_matmul.out — same product, written into the caller's buffer."""
    logger.debug("GEMS_ENFLAME CHAIN_MATMUL_OUT")
    return run(matrices, *args, out=out)
