# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 _foreach_mul / _foreach_mul_（全部 4 种 overload）。

公共 foreach 执行器用二维 grid 承载 tensor 列表，GCU400 的 ``grid.y``
上限 255 在大 shape 下会 ``out of resource``。浮点连续列表改走
``_foreach_flat`` 的一维固定 grid / metadata 融合 kernel；非浮点、
非连续或广播等形态仍委托公共执行器保持 dtype policy 与异常语义。

融合 kernel 的 metadata 上传（``torch.tensor(..., device=)``）是每次
调用 ~90us 的同步 H2D，让所有不带缓存的 overload 的延迟恒定在
shape 无关的 ~100-200us。四段 metadata 只含地址与元素数，kernel
每次仍完整读取张量数据，按 ``(device, 地址, 元素数)`` 精确缓存后在
稳态调用（分配器复用同一地址）中零 H2D；ScalarList 的 fp32 标量表
同样只读，独立按值缓存。
"""

import logging

import torch

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar, _apply_tensor
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    MODE_TENSOR,
    OP_MUL,
    _fused_grid,
    can_use_flat,
    fused_binary_list_kernel,
    fused_binary_scalar_kernel,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Generated per round by the optimization driver script; all compile-time / launch-time config, semantics unchanged.
FIXED_GRID = 48
BLOCK_SIZE = 8192
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# All overloads reuse the read-only _device_meta cache, eliminating the synchronous H2D on steady-state calls.
USE_META_CACHE = 1
# The fp32 scalar table for .ScalarList is cached separately by (device, value).
USE_SCALAR_CACHE = 1
SCALAR_CACHE_LIMIT = 64
# Small shapes use the common binary executor (its 2D grid stays in range and its cache is more complete).
PUBLIC_SMALL_SHAPE = 0
SMALL_SHAPE_MAX_NUMEL = 261120
# --- TUNING PARAMS END ---

# Read-only metadata cache: the key contains only device and address/element count, not tensor data; the kernel still
# fully reads inputs and recomputes each time, so address reuse or in-place content changes do not affect correctness.
_SCALAR_CACHE = {}


def _cached_scalar_tensor(scalars, device):
    """ScalarList 的 fp32 标量表（只读），按 (device, 值元组) 缓存。"""
    key = (str(device), tuple(scalars))
    cached = _SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_SCALAR_CACHE) >= SCALAR_CACHE_LIMIT:
        _SCALAR_CACHE.clear()
    cached = torch.tensor(scalars, dtype=torch.float32, device=device)
    _SCALAR_CACHE[key] = cached
    return cached


def _run_mul_cached(self, operand, mode, inplace):
    """四种操作数形态的精简单 launch 路径（metadata 走只读缓存）。

    不适用的形态返回 ``FALLBACK``，由调用方决定回退目标：广播 ``.List``
    需要 ``run_binary`` 的逐 tensor kernel；其余形态回退公共执行器。
    metadata 的四段布局（输入/输出/第二操作数地址 + 各元素数）与
    ``_foreach_flat`` 的融合 kernel 完全一致。
    """
    if not can_use_flat(self, operand, mode):
        return FALLBACK
    num_tensors = len(self)
    in_ptrs = [tensor.data_ptr() for tensor in self]
    outputs = self if inplace else [torch.empty_like(tensor) for tensor in self]
    out_ptrs = in_ptrs if inplace else [tensor.data_ptr() for tensor in outputs]
    numels = [tensor.numel() for tensor in self]
    dtype = self[0].dtype
    triton_dtype = _TL_DTYPES[dtype]
    device = self[0].device
    grid = (_fused_grid(self, FIXED_GRID, BLOCK_SIZE),)

    if mode == MODE_LIST:
        if not all(
            other.numel() == tensor.numel() for tensor, other in zip(self, operand)
        ):
            # Broadcast .List does not share flat indices with the inputs, so hand it to the per-tensor kernel.
            return FALLBACK
        b_ptrs = [other.data_ptr() for other in operand]
        meta = _device_meta(in_ptrs + out_ptrs + b_ptrs + numels, device)
        fused_binary_list_kernel[grid](
            meta,
            num_tensors,
            OP_MUL,
            triton_dtype,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    elif mode == MODE_SCALAR_LIST:
        if USE_SCALAR_CACHE:
            scalar_tensor = _cached_scalar_tensor(operand, device)
        else:
            scalar_tensor = torch.tensor(operand, dtype=torch.float32, device=device)
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        fused_binary_scalar_kernel[grid](
            meta,
            scalar_tensor,
            num_tensors,
            OP_MUL,
            triton_dtype,
            0,
            triton_dtype,
            0.0,
            1,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    elif mode == MODE_TENSOR:
        # A 0-d tensor is often fp32; it must be read using its own dtype, not the list dtype.
        b_dtype = _TL_DTYPES.get(operand.dtype, triton_dtype)
        meta = _device_meta(
            in_ptrs + out_ptrs + [operand.data_ptr()] * num_tensors + numels, device
        )
        fused_binary_scalar_kernel[grid](
            meta,
            operand,
            num_tensors,
            OP_MUL,
            triton_dtype,
            1,
            b_dtype,
            0.0,
            0,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    else:
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        fused_binary_scalar_kernel[grid](
            meta,
            self[0],
            num_tensors,
            OP_MUL,
            triton_dtype,
            0,
            triton_dtype,
            float(operand),
            0,
            FP32_PROMOTE,
            BLOCK_SIZE,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    return None if inplace else outputs


def _dispatch(self, operand, mode, inplace, fallback):
    """精简单 launch 路径优先；广播/异常形态回退公共语义。"""
    if FALLBACK_TO_PUBLIC:
        return fallback()
    if USE_META_CACHE:
        result = _run_mul_cached(self, operand, mode, inplace)
        if result is not FALLBACK:
            return result
        if mode != MODE_LIST:
            # can_use_flat failed: leave dtype policy and exception semantics to the common executor.
            return fallback()
    result = run_binary(
        self,
        operand,
        OP_MUL,
        mode,
        inplace,
        fixed_grid=FIXED_GRID,
        block_size=BLOCK_SIZE,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        use_device_context=USE_DEVICE_CONTEXT,
        fp32_promote=FP32_PROMOTE,
        fused=FUSED_LAUNCH,
    )
    if result is FALLBACK:
        return fallback()
    return result


def _foreach_mul_List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALAR")
    return _dispatch(
        self, other, MODE_LIST, False, lambda: _apply_list("mul", self, other, False)
    )


def _foreach_mul_Scalar(self, scalar):
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        False,
        lambda: _apply_scalar("mul", self, scalar, False),
    )


def _foreach_mul_ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        False,
        lambda: _apply_scalar("mul", self, scalars, False),
    )


def _foreach_mul_Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL_TENSOR")
    return _dispatch(
        self,
        other,
        MODE_TENSOR,
        False,
        lambda: _apply_tensor("mul", self, other, False),
    )


def _foreach_mul__List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__LIST")
    return _dispatch(
        self, other, MODE_LIST, True, lambda: _apply_list("mul", self, other, True)
    )


def _foreach_mul__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALAR")
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        True,
        lambda: _apply_scalar("mul", self, scalar, True),
    )


def _foreach_mul__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        True,
        lambda: _apply_scalar("mul", self, scalars, True),
    )


def _foreach_mul__Tensor(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MUL__TENSOR")
    return _dispatch(
        self, other, MODE_TENSOR, True, lambda: _apply_tensor("mul", self, other, True)
    )
