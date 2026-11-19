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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

_INT32_MAX = torch.iinfo(torch.int32).max
# Element-wise Triton stores of 64-bit dtypes are not reliable on this backend
# (the runtime replaces Long/Double with Int, corrupting every other element),
# so they stay on the generic broadcast kernel.  Complex lowering is just as
# fragile on this backend (see new_full).
_FLAT_UNSAFE_DTYPES = (
    torch.int64,
    torch.uint64,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
# GCU400 tuning for pure-write streams (new_full RSI, 2026-11-16): grids above
# the 24 schedulable SIPs never help pure writes (48 == 24) and 32 misfits the
# SIP count (~+20% on mid shapes); 32Ki-element tiles are the mid-band sweet
# spot and tie the ~3.25 TB/s DRAM write wall at stream sizes.
_FLAT_MAX_GRID = 24
_FLAT_MIN_BLOCK = 2048
_FLAT_SWEET_BLOCK = 32768
_LAUNCH_ATTEMPTS = 3


@pointwise_dynamic(
    is_tensor=[True, False], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_scalar_func(inp, value_scalar):
    return tl.full(inp.shape, value_scalar, dtype=inp.dtype)


@pointwise_dynamic(
    is_tensor=[True, True], promotion_methods=[(0, "DEFAULT")], num_outputs=1
)
@triton.jit
def fill_tensor_func(inp, value):
    return value


@libentry()
@triton.jit
def fill_scalar_flat_kernel(
    out_ptr,
    fill_value,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The fill value is a plain Python scalar: write the broadcast constant in
    # flat memory order with a grid-stride loop capped at the schedulable SIPs.
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, val)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, val, mask=mask)


@libentry()
@triton.jit
def fill_tensor_flat_kernel(
    out_ptr,
    value_ptr,
    n_elements,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    # The value is a 0-dim device tensor: load it once per CTA and broadcast
    # it on the store, instead of materializing a stride-0 block load per tile.
    fill_value = tl.load(value_ptr)
    pid = tl.program_id(0)
    num_ctas = tl.num_programs(0)
    n_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, n_tiles, num_ctas):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        val = tl.full([BLOCK], fill_value, out_ptr.dtype.element_ty)
        if EXACT:
            tl.store(out_ptr + offsets, val)
        else:
            mask = offsets < n_elements
            tl.store(out_ptr + offsets, val, mask=mask)


def _flat_block(n_elements):
    if n_elements <= 8192:
        # Small sizes sit on the launch floor: one CTA with an exact tile.
        return max(2, triton.next_power_of_2(n_elements))
    # 32Ki-element tiles are the mid-band sweet spot (ramp-friendly: -1.4%~-4%
    # vs 64/128Ki caps at 4-64MB) and tie the DRAM write wall at stream sizes
    # (BLOCK is not the limiter once the 24 SIPs saturate writes).
    return min(
        _FLAT_SWEET_BLOCK,
        max(_FLAT_MIN_BLOCK, triton.next_power_of_2(-(-n_elements // 48))),
    )


def _launch_flat(kernel, store_target, args, n_elements, num_warps=2):
    block = _flat_block(n_elements)
    grid = (min(_FLAT_MAX_GRID, -(-n_elements // block)),)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level flake windows on shared nodes); retrying
    # the launch recompiles and succeeds (new_full / atan2_ / sinh precedents
    # for the same toolchain family).
    for attempt in range(_LAUNCH_ATTEMPTS):
        try:
            with torch_device_fn.device(store_target.device):
                kernel[grid](
                    store_target,
                    *args,
                    n_elements,
                    BLOCK=block,
                    EXACT=(n_elements % block == 0),
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == _LAUNCH_ATTEMPTS - 1:
                raise
            logger.warning(
                "fill flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _flat_store_target(out):
    if out.dtype == torch.bool:
        # Store through a same-width int8 view: bool element stores lower to
        # the narrow i1 lane on this backend.
        return out.view(torch.int8)
    return out


def _fill_scalar_flat_eligible_fresh(input, out):
    # ``out`` is freshly allocated by ``torch.empty_like`` so it always owns a
    # dense, non-overlapping storage; a cheap layout check replaces the aten
    # dense-query (which costs a dispatcher hop on every call).
    if input.layout != torch.strided:
        return False
    if out.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    return out.numel() <= _INT32_MAX


def _fill_scalar_flat_eligible_into(input, out):
    if out.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    if out.numel() > _INT32_MAX:
        return False
    if out.shape != input.shape:
        return False
    # A uniform fill writes the same value to every element, so any dense
    # non-overlapping layout can be filled in flat memory order.
    return torch.ops.aten.is_non_overlapping_and_dense(out)


def _fill_scalar_flat(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return
    store_target = _flat_store_target(out)
    if out.dtype == torch.bool:
        value = 1 if value else 0
    _launch_flat(fill_scalar_flat_kernel, store_target, (value,), n_elements)


def _fill_tensor_flat_eligible_fresh(input, out, value):
    if not _fill_scalar_flat_eligible_fresh(input, out):
        return False
    if value.dtype in _FLAT_UNSAFE_DTYPES or value.dtype != out.dtype:
        return False
    return True


def _fill_tensor_flat_eligible_into(input, out, value):
    if out.dtype in _FLAT_UNSAFE_DTYPES or value.dtype in _FLAT_UNSAFE_DTYPES:
        return False
    if value.dtype != out.dtype:
        return False
    if out.numel() > _INT32_MAX:
        return False
    if out.shape != input.shape:
        return False
    return torch.ops.aten.is_non_overlapping_and_dense(out)


def _fill_tensor_flat(out, value):
    n_elements = out.numel()
    if n_elements == 0:
        return
    store_target = _flat_store_target(out)
    if out.dtype == torch.bool:
        value = value.view(torch.int8)
    _launch_flat(fill_tensor_flat_kernel, store_target, (value,), n_elements)


def fill_scalar(input, value):
    logger.debug("GEMS_ENFLAME FILL")
    out = torch.empty_like(input)
    if _fill_scalar_flat_eligible_fresh(input, out):
        _fill_scalar_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        return fill_scalar_func(input, value, out0=out)


def fill_tensor(input, value):
    if not value.is_cuda:
        return fill_scalar(input, value.item())
    logger.debug("GEMS_ENFLAME FILL")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    out = torch.empty_like(input)
    if _fill_tensor_flat_eligible_fresh(input, out, value):
        _fill_tensor_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        return fill_tensor_func(input, value, out0=out)


def fill_scalar_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_OUT")
    if out is None:
        return fill_scalar(input, value)
    if _fill_scalar_flat_eligible_into(input, out):
        _fill_scalar_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        fill_scalar_func(input, value, out0=out)
    return out


def fill_tensor_out(input, value, *, out=None):
    logger.debug("GEMS_ENFLAME FILL_TENSOR_OUT")
    if out is None:
        return fill_tensor(input, value)
    if not value.is_cuda:
        return fill_scalar_out(input, value.item(), out=out)
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible_into(input, out, value):
        _fill_tensor_flat(out, value)
        return out
    with torch_device_fn.device(input.device):
        fill_tensor_func(input, value, out0=out)
    return out


def fill_tensor_(self, value):
    if not value.is_cuda:
        return fill_scalar_(self, value.item())
    logger.debug("GEMS_ENFLAME FILL_TENSOR_")
    if value.ndim != 0:
        raise RuntimeError(
            f"fill_ only supports 0-dimension value tensor but got tensor with {value.ndim} dimensions."
        )
    if _fill_tensor_flat_eligible_into(self, self, value):
        _fill_tensor_flat(self, value)
        return self
    with torch_device_fn.device(self.device):
        fill_tensor_func(self, value, out0=self)
    return self


def fill_scalar_(self, value=0):
    logger.debug("GEMS_ENFLAME FILL_SCALAR_")
    if _fill_scalar_flat_eligible_into(self, self):
        _fill_scalar_flat(self, value)
        return self
    with torch_device_fn.device(self.device):
        fill_scalar_func(self, value, out0=self)
    return self
