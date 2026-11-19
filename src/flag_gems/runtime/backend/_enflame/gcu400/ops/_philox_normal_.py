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

# GCU400 specialization of ``aten::_philox_normal_``.
#
# The generic launch fixes BLOCK=512 and a 1D grid of cdiv(N, BLOCK*4)
# programs, which overflows the 65535-program grid.x hard limit of this
# backend for the 2**30-element benchmark shapes (grid = 524288 ->
# ``OutOfResources: out of resource: grid.x``).  This copy keeps the ATen
# semantics bit-exact (the accuracy suite compares against a NumPy
# re-implementation of ``PhiloxDistribution.cu``, so the Philox-4x32-10
# chain and the Box-Muller pairing must not change) and reworks only the
# launch shape and the store path, following the sealed ``_philox_uniform_``
# recipe (job j20261118014844_ae36db10):
#
#   * One persistent grid-stride kernel covers the whole tensor with at
#     most 48 programs, so the grid never depends on the problem size and
#     the grid.x cap is structurally unreachable; small tensors keep the
#     one-block-per-program shape where the persistent loop would cost
#     ~0.4us at the launch floor.
#   * The 32-bit counter fast path is selected by a runtime scalar branch
#     inside the kernel (the (seed, offset) pair lives in a device tensor;
#     reading it on the host would sync every call): when
#     ``offset + total_counters <= 2**32`` and ``n < 2**31`` the counters
#     and the flat offsets stay in uint32; otherwise the generic 64-bit
#     counter math runs (large offsets, e.g. ``2**40``, stay exact).
#   * 16-bit outputs pack two samples into one uint32 store (PACK2): lane
#     ``k`` writes word ``2k`` as ``v0 | v1 << 16`` and word ``2k+1`` as
#     ``v2 | v3 << 16`` -- ATen's element order ``out[4k + j] = n_j`` maps
#     to exactly these two words. Odd sizes keep the plain stores.
#   * EVEN drops the store masks when ``n % (BLOCK * 4) == 0``.
#
# The Philox chain itself stays on ``tl.philox``: a hand-written round that
# shares the 32x32 products probed no faster at BLOCK=1024 (332.4 vs
# 317.4 us at 2**24 fp32) and the u32-carry offset variant exposed a
# device-level fault axis (Sip exception -> KMD abort at fp32 x 2**30, see
# compiler_problems/P16 sources/flaggems-enflame__philox_normal_.md), so
# both were rejected in favor of the sealed uniform-family counter scheme.
#
# Host contract (2026-11-12 ruling): the wrapper only validates arguments,
# allocates the staging buffer for non-contiguous outputs and launches these
# kernels; no aten compute op is called.
#
# This module also publishes itself over the generic package attribute
# ``flag_gems.ops._philox_normal_`` (the same vendor-injection pattern as
# ``conj_physical`` / ``glu_backward_jvp``): SpecOpRegistrar rebinds the
# top-level ``flag_gems._philox_normal_`` and the aten registration, but
# the perf benchmark resolves the op through the submodule attribute, which
# otherwise keeps pointing at the generic implementation -- the one whose
# grid computation cannot run the core shapes at all.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# `aten::_philox_normal_` is part of PyTorch's stateless Philox RNG family,
# which landed upstream after the 2.10 release. Builds that predate it ship no
# schema for the operator, and FlagGems' registrar refuses to install an
# implementation for an unknown schema. Define the upstream signature verbatim
# when it is missing so the operator can still be registered and exercised;
# on newer torch builds the guard short-circuits.
_PHILOX_NORMAL_SCHEMA = (
    "(Tensor(a!) self, Tensor key, float mean=0, float std=1) -> Tensor(a!)"
)


def _ensure_philox_normal_schema():
    if hasattr(torch.ops.aten, "_philox_normal_"):
        return
    try:
        torch.library.define("aten::_philox_normal_", _PHILOX_NORMAL_SCHEMA)
    except Exception as exc:  # pragma: no cover - depends on the torch build
        logger.warning("Unable to define aten::_philox_normal_: %s", exc)


_ensure_philox_normal_schema()

# Launch ladder (do_bench kernel mode, median w1000/r100, GCU400/L600):
# below 16K elements every configuration sits at the ~4.2us launch floor, so
# the one-block-per-program shape wins; from 2**24 elements on, persistent
# grid-stride with BLOCK=2048 / 48 programs / num_warps=4 is the sealed
# uniform-family sweet spot, re-confirmed for the Box-Muller variant by the
# probe sweep in this job (see case log).
_SMALL_MAX_NUMEL = 1 << 14
_SMALL_BLOCK = 1024
_SMALL_NUM_WARPS = 2
_PERSIST_BLOCK = 2048
_PERSIST_NUM_PROG = 48
_PERSIST_NUM_WARPS = 4
# Two 16-bit samples per uint32 store only pay off once the kernel is past
# the launch floor; smaller or odd-sized requests keep the plain stores.
_PACK2_MIN_NUMEL = 8192
_F64_BLOCK = 512
_MAX_GRID_X = 65535


@triton.jit
def _box_muller_f32(r0, r1, r2, r3):
    """Pair four Philox words into four standard normals, in float32.

    Mirrors ``box_muller_float`` from cuda/PhiloxDistribution.cu: each word
    becomes a uniform in (0, 1] through the fused multiply-add
    ``fmaf(word, 2**-32, 2**-33)``, odd-indexed words turn into angles and
    even-indexed ones into radii, and words 0/1 and 2/3 form the two
    (cosine, sine) pairs. Both constants are powers of two, so the mapping is
    exact.
    """
    u1 = tl.fma(r0.to(tl.float32), 2.3283064365386963e-10, 1.1641532182693481e-10)
    u2 = tl.fma(r1.to(tl.float32), 2.3283064365386963e-10, 1.1641532182693481e-10)
    u3 = tl.fma(r2.to(tl.float32), 2.3283064365386963e-10, 1.1641532182693481e-10)
    u4 = tl.fma(r3.to(tl.float32), 2.3283064365386963e-10, 1.1641532182693481e-10)

    radius1 = tl.sqrt(-2.0 * tl.log(u1))
    radius2 = tl.sqrt(-2.0 * tl.log(u3))
    theta1 = 6.2831853071795864 * u2
    theta2 = 6.2831853071795864 * u4

    return (
        radius1 * tl.cos(theta1),
        radius1 * tl.sin(theta1),
        radius2 * tl.cos(theta2),
        radius2 * tl.sin(theta2),
    )


@triton.jit
def _box_muller_f64(r0, r1, r2, r3):
    """Pair four Philox words into two standard normals, in float64.

    Mirrors ``box_muller_double``: each normal spends two words, packed into
    roughly 64 bits of uniform randomness instead of 32, so one Philox call
    yields two samples rather than four.
    """
    m = 2.3283064365386963e-10
    u1 = tl.fma(r0.to(tl.float64), m, r1.to(tl.float64) * m * m + m * m * 0.5)
    u2 = tl.fma(r2.to(tl.float64), m, r3.to(tl.float64) * m * m + m * m * 0.5)

    radius = tl.sqrt(-2.0 * tl.log(u1))
    theta = 6.2831853071795864 * u2

    return radius * tl.cos(theta), radius * tl.sin(theta)


@triton.jit(do_not_specialize=["n_elem", "num_prog"])
def _philox_normal_gcu400(
    out_ptr,
    key_ptr,
    n_elem,
    mean,
    std,
    num_prog,
    BLOCK: tl.constexpr,
    PACK2: tl.constexpr,
    EVEN: tl.constexpr,
    PERSIST: tl.constexpr,
):
    """Fill ``out_ptr`` with normal samples from a stateless Philox stream.

    Element ``i`` comes from the ``i // 4``-th Philox 4x32 counter value
    (counter = offset + i // 4, subsequence zero), word ``i % 4``, mapped
    through ATen's Box-Muller transform. The 32-bit fast path is taken per
    launch via a runtime scalar branch; both branches produce identical
    values for the same (seed, offset).
    """
    seed = tl.load(key_ptr).to(tl.uint64)
    offset = tl.load(key_ptr + 1).to(tl.uint64)
    pid = tl.program_id(0)
    total_blocks = (n_elem + BLOCK * 4 - 1) // (BLOCK * 4)
    # uint32 fast path: the whole counter window and the flat element
    # offsets must stay inside 32 bits. Checked on the device because the
    # (seed, offset) pair lives in a device tensor; a host-side check would
    # sync every call. offset < 2**32 is implied by the first bound.
    u32_ok = (n_elem < 0x80000000) & (
        offset + total_blocks.to(tl.uint64) * BLOCK <= 0x100000000
    )
    dtype = out_ptr.dtype.element_ty
    if PERSIST:
        if u32_ok:
            for block_idx in range(pid, total_blocks, num_prog):
                chunks = block_idx * BLOCK + tl.arange(0, BLOCK).to(tl.uint32)
                c0 = offset.to(tl.uint32) + chunks
                z = c0 * 0
                r0, r1, r2, r3 = tl.philox(seed, c0, z, z, z)
                n0, n1, n2, n3 = _box_muller_f32(r0, r1, r2, r3)
                v0 = tl.fma(n0, std, mean).to(dtype)
                v1 = tl.fma(n1, std, mean).to(dtype)
                v2 = tl.fma(n2, std, mean).to(dtype)
                v3 = tl.fma(n3, std, mean).to(dtype)
                if PACK2:
                    u0 = v0.to(tl.uint16, bitcast=True).to(tl.uint32)
                    u1 = v1.to(tl.uint16, bitcast=True).to(tl.uint32)
                    u2 = v2.to(tl.uint16, bitcast=True).to(tl.uint32)
                    u3 = v3.to(tl.uint16, bitcast=True).to(tl.uint32)
                    w0 = u0 | (u1 << 16)
                    w1 = u2 | (u3 << 16)
                    wbase = chunks * 2
                    words_ptr = out_ptr.to(tl.pointer_type(tl.uint32))
                    if EVEN:
                        tl.store(words_ptr + wbase, w0)
                        tl.store(words_ptr + wbase + 1, w1)
                    else:
                        n_words = (n_elem // 2).to(tl.uint32)
                        tl.store(words_ptr + wbase, w0, mask=wbase < n_words)
                        tl.store(words_ptr + wbase + 1, w1, mask=wbase + 1 < n_words)
                else:
                    base = chunks * 4
                    if EVEN:
                        tl.store(out_ptr + base, v0)
                        tl.store(out_ptr + base + 1, v1)
                        tl.store(out_ptr + base + 2, v2)
                        tl.store(out_ptr + base + 3, v3)
                    else:
                        n32 = n_elem.to(tl.uint32)
                        tl.store(out_ptr + base, v0, mask=base < n32)
                        tl.store(out_ptr + base + 1, v1, mask=base + 1 < n32)
                        tl.store(out_ptr + base + 2, v2, mask=base + 2 < n32)
                        tl.store(out_ptr + base + 3, v3, mask=base + 3 < n32)
        else:
            for block_idx in range(pid, total_blocks, num_prog):
                chunks64 = block_idx.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(
                    tl.int64
                )
                counter = offset + chunks64.to(tl.uint64)
                c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
                c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
                z = c0 * 0
                r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
                n0, n1, n2, n3 = _box_muller_f32(r0, r1, r2, r3)
                base64 = chunks64 * 4
                tl.store(
                    out_ptr + base64,
                    tl.fma(n0, std, mean).to(dtype),
                    mask=base64 < n_elem,
                )
                tl.store(
                    out_ptr + base64 + 1,
                    tl.fma(n1, std, mean).to(dtype),
                    mask=base64 + 1 < n_elem,
                )
                tl.store(
                    out_ptr + base64 + 2,
                    tl.fma(n2, std, mean).to(dtype),
                    mask=base64 + 2 < n_elem,
                )
                tl.store(
                    out_ptr + base64 + 3,
                    tl.fma(n3, std, mean).to(dtype),
                    mask=base64 + 3 < n_elem,
                )
    else:
        if u32_ok:
            chunks = pid * BLOCK + tl.arange(0, BLOCK).to(tl.uint32)
            c0 = offset.to(tl.uint32) + chunks
            z = c0 * 0
            r0, r1, r2, r3 = tl.philox(seed, c0, z, z, z)
            n0, n1, n2, n3 = _box_muller_f32(r0, r1, r2, r3)
            v0 = tl.fma(n0, std, mean).to(dtype)
            v1 = tl.fma(n1, std, mean).to(dtype)
            v2 = tl.fma(n2, std, mean).to(dtype)
            v3 = tl.fma(n3, std, mean).to(dtype)
            if PACK2:
                u0 = v0.to(tl.uint16, bitcast=True).to(tl.uint32)
                u1 = v1.to(tl.uint16, bitcast=True).to(tl.uint32)
                u2 = v2.to(tl.uint16, bitcast=True).to(tl.uint32)
                u3 = v3.to(tl.uint16, bitcast=True).to(tl.uint32)
                w0 = u0 | (u1 << 16)
                w1 = u2 | (u3 << 16)
                wbase = chunks * 2
                words_ptr = out_ptr.to(tl.pointer_type(tl.uint32))
                if EVEN:
                    tl.store(words_ptr + wbase, w0)
                    tl.store(words_ptr + wbase + 1, w1)
                else:
                    n_words = (n_elem // 2).to(tl.uint32)
                    tl.store(words_ptr + wbase, w0, mask=wbase < n_words)
                    tl.store(words_ptr + wbase + 1, w1, mask=wbase + 1 < n_words)
            else:
                base = chunks * 4
                if EVEN:
                    tl.store(out_ptr + base, v0)
                    tl.store(out_ptr + base + 1, v1)
                    tl.store(out_ptr + base + 2, v2)
                    tl.store(out_ptr + base + 3, v3)
                else:
                    n32 = n_elem.to(tl.uint32)
                    tl.store(out_ptr + base, v0, mask=base < n32)
                    tl.store(out_ptr + base + 1, v1, mask=base + 1 < n32)
                    tl.store(out_ptr + base + 2, v2, mask=base + 2 < n32)
                    tl.store(out_ptr + base + 3, v3, mask=base + 3 < n32)
        else:
            chunks64 = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
            counter = offset + chunks64.to(tl.uint64)
            c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
            c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
            z = c0 * 0
            r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
            n0, n1, n2, n3 = _box_muller_f32(r0, r1, r2, r3)
            base64 = chunks64 * 4
            tl.store(
                out_ptr + base64, tl.fma(n0, std, mean).to(dtype), mask=base64 < n_elem
            )
            tl.store(
                out_ptr + base64 + 1,
                tl.fma(n1, std, mean).to(dtype),
                mask=base64 + 1 < n_elem,
            )
            tl.store(
                out_ptr + base64 + 2,
                tl.fma(n2, std, mean).to(dtype),
                mask=base64 + 2 < n_elem,
            )
            tl.store(
                out_ptr + base64 + 3,
                tl.fma(n3, std, mean).to(dtype),
                mask=base64 + 3 < n_elem,
            )


@triton.jit(do_not_specialize=["n_elem", "num_prog"])
def _philox_normal_f64_gcu400(
    out_ptr,
    key_ptr,
    params_ptr,
    n_elem,
    num_prog,
    BLOCK: tl.constexpr,
):
    """float64 arm: two samples per counter, 64-bit counter math, grid-stride.

    ``params_ptr`` addresses a ``(mean, std)`` pair of doubles. They arrive as
    a device tensor rather than Python floats because Triton narrows float
    kernel arguments to float32, which would drop the low mantissa bits of an
    arbitrary mean/std. Kept for ATen parity; gcu400 has no fp64 arithmetic,
    so this kernel is never exercised by the accuracy suite on this target
    (fp64 is filtered from the test dtypes).
    """
    seed = tl.load(key_ptr).to(tl.uint64)
    offset = tl.load(key_ptr + 1).to(tl.uint64)
    mean = tl.load(params_ptr)
    std = tl.load(params_ptr + 1)
    pid = tl.program_id(0)
    total_blocks = (n_elem + BLOCK * 2 - 1) // (BLOCK * 2)
    for block_idx in range(pid, total_blocks, num_prog):
        chunks = block_idx.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        counter = offset + chunks.to(tl.uint64)
        c0 = (counter & 0xFFFFFFFF).to(tl.uint32)
        c1 = ((counter >> 32) & 0xFFFFFFFF).to(tl.uint32)
        z = c0 * 0
        r0, r1, r2, r3 = tl.philox(seed, c0, c1, z, z)
        n0, n1 = _box_muller_f64(r0, r1, r2, r3)
        base = chunks * 2
        tl.store(out_ptr + base, tl.fma(n0, std, mean), mask=base < n_elem)
        tl.store(out_ptr + base + 1, tl.fma(n1, std, mean), mask=base + 1 < n_elem)


def _launch(out, key, mean, std):
    n = out.numel()
    if out.dtype == torch.float64:
        # ATen accumulates 8-byte outputs in double, so the parameters are
        # cast to double before they reach the kernel.
        params = torch.tensor([mean, std], dtype=torch.float64, device=out.device)
        # Two elements per Philox call, so each lane covers half as much as in
        # the 16/32-bit path.
        grid = min(triton.cdiv(n, _F64_BLOCK * 2), _MAX_GRID_X)
        with torch_device_fn.device(out.device):
            _philox_normal_f64_gcu400[(grid,)](
                out,
                key,
                params,
                n,
                grid,
                BLOCK=_F64_BLOCK,
                num_warps=4,
            )
        return

    is_16bit = out.dtype in (torch.float16, torch.bfloat16)
    pack2 = is_16bit and n % 2 == 0 and n >= _PACK2_MIN_NUMEL
    if n <= _SMALL_MAX_NUMEL:
        block = _SMALL_BLOCK
        grid = triton.cdiv(n, block * 4)
        with torch_device_fn.device(out.device):
            _philox_normal_gcu400[(grid,)](
                out,
                key,
                n,
                mean,
                std,
                grid,
                BLOCK=block,
                PACK2=pack2,
                EVEN=n % (block * 4) == 0,
                PERSIST=False,
                num_warps=_SMALL_NUM_WARPS,
            )
        return

    block = _PERSIST_BLOCK
    grid = min(triton.cdiv(n, block * 4), _PERSIST_NUM_PROG)
    with torch_device_fn.device(out.device):
        _philox_normal_gcu400[(grid,)](
            out,
            key,
            n,
            mean,
            std,
            grid,
            BLOCK=block,
            PACK2=pack2,
            EVEN=n % (block * 4) == 0,
            PERSIST=True,
            num_warps=_PERSIST_NUM_WARPS,
        )


def _check_args(self, key):
    if not self.is_floating_point():
        raise RuntimeError(
            f"_philox_normal_: self must be a floating point tensor, got {self.dtype}"
        )
    if key.dtype != torch.uint64:
        raise RuntimeError(
            f"_philox_normal_: key must have dtype uint64, got {key.dtype}"
        )
    if self.device != key.device:
        raise RuntimeError(
            f"_philox_normal_: self and key must be on the same device, got "
            f"{self.device} and {key.device}"
        )
    if key.dim() < 1 or key.size(-1) != 2:
        raise RuntimeError(
            f"_philox_normal_: key must have shape (2,) or (*batch, 2), got "
            f"{tuple(key.shape)}"
        )
    if key.dim() > 1 and key.dim() != self.dim() + 1:
        raise RuntimeError(
            f"_philox_normal_: batched key must have ndim == output ndim + 1, got "
            f"key shape {tuple(key.shape)} with output shape {tuple(self.shape)}"
        )


def _batched_key_plan(out, key):
    """Split a batched key into per-key output blocks and key indices.

    ATen derives the elements one key owns by counting the output's trailing
    dimensions that the key leaves unsized (``key.size(i) == 1``); a key with
    shape ``(4, 1, 1, 2)`` against output ``(4, 10, 100)`` therefore feeds 1000
    elements. The remaining key dimensions broadcast, so the key index for a
    block is the flat index of a zero-based multi-index along those dims.
    """
    elems_per_key = 1
    key_dims = out.dim()
    for i in range(out.dim() - 1, -1, -1):
        if key.size(i) != 1:
            break
        elems_per_key *= out.size(i)
        key_dims -= 1

    key_batch = tuple(key.shape[:key_dims])
    out_batch = tuple(out.shape[:key_dims])
    for ks, os_ in zip(key_batch, out_batch):
        if ks != os_ and ks != 1:
            raise RuntimeError(
                f"_philox_normal_: key batch shape {key_batch} is not "
                f"broadcastable with output shape {out_batch}"
            )

    num_blocks = out.numel() // elems_per_key
    # Walk the batch dims in output order, advancing the key's flat index only
    # where the key is actually sized (broadcast dims keep their index).
    strides = []
    acc = 1
    for ks in reversed(key_batch):
        strides.append(0 if ks == 1 else acc)
        acc *= ks
    strides.reverse()

    block = []
    for b in range(num_blocks):
        idx = 0
        rem = b
        for dim in range(key_dims - 1, -1, -1):
            size = out_batch[dim]
            coord = rem % size if size else 0
            rem //= size if size else 1
            idx += coord * strides[dim]
        block.append(idx)
    return elems_per_key, block


def _philox_normal_(self, key, mean=0.0, std=1.0):
    logger.debug("GEMS _PHILOX_NORMAL_")
    _check_args(self, key)
    if self.numel() == 0:
        return self

    key = key.contiguous()
    # Non-contiguous outputs are filled through a contiguous staging buffer and
    # copied back, so the kernel can rely on plain linear indexing.
    contiguous = self.is_contiguous()
    out = self if contiguous else self.contiguous()

    if key.dim() == 1:
        _launch(out, key, mean, std)
    else:
        elems_per_key, block_keys = _batched_key_plan(out, key)
        key_flat = key.reshape(-1, 2)
        out_flat = out.reshape(-1, elems_per_key)
        for block_idx, key_idx in enumerate(block_keys):
            _launch(out_flat[block_idx], key_flat[key_idx], mean, std)

    if not contiguous:
        self.copy_(out)
    return self


# ---------------------------------------------------------------------------
# Namespace sync: ``SpecOpRegistrar`` only rebinds the top-level ``flag_gems``
# namespace, so the ``flag_gems.ops._philox_normal_`` submodule attribute --
# the object benchmark/test_philox_normal_.py binds as ``gems_op`` -- keeps
# pointing at the generic implementation, whose 1D cdiv(N, 2048) grid
# overflows the 65535-program grid.x cap on the 2**30 core shapes
# (``OutOfResources: out of resource: grid.x``, P19).  Rebind the package
# attribute to this specialized host so every reference path (top level,
# ATen registration, ops submodule) runs the GCU400 kernel.  ``flag_gems.ops``
# is fully imported before the registrar loads this module, so the import
# below is a plain sys.modules hit.
try:
    import flag_gems.ops as _generic_ops_pkg
except ImportError:  # pragma: no cover - flag_gems.ops always importable
    _generic_ops_pkg = None

if (
    _generic_ops_pkg is not None
    and getattr(_generic_ops_pkg, "_philox_normal_", None) is not _philox_normal_
):
    _generic_ops_pkg._philox_normal_ = _philox_normal_
