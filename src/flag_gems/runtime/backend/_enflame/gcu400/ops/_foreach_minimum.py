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

"""GCU400-specific scalar overloads of _foreach_minimum / _foreach_minimum_.

The common foreach executor carries the tensor list in a 2D grid, and GCU400's ``grid.y``
limit of 255 causes ``out of resource`` on large shapes. Contiguous float lists use the
1D fixed grid / metadata fused kernel from ``_foreach_flat``; non-float,
non-contiguous or broadcast shapes still delegate to the common executor to keep dtype policy and exception semantics.

The paired ``.List`` path runs its own single-launch kernel (``_minimum_list_kernel``):
same four-segment int64 metadata and ``(tensor, part)`` linearization as the common
``fused_binary_list_kernel``, with two GCU400-specific refinements measured on the
official do_bench kernel protocol (fresh stack re-anchor, jobs/j20261119081737):

* the grid is an exact multiple of the tensor count by construction (see
  ``_minimum_grid``), so every tile of every tensor is covered for any list
  length -- ``_foreach_flat._fused_grid`` caps by ``min(fixed_grid, NT*tiles)``
  and left the tiles of the trailing ``grid % NT`` tensors unvisited for list
  lengths that do not divide the cap;
* ``EXACT`` drops the per-tile bounds mask when every element count divides
  the chosen ``BLOCK``, and the block is tiered by dtype: fp32 stays at 8192,
  16-bit moves to 16384. On this stack EXACT@8192 wins for fp32 (-17/-11/-5%
  at 256^2/512^2/1024^2) while for bf16 the same 8192 tile *regresses*
  (+2~3%) and only the 16384 tile turns the mask removal into a win -- the
  (dtype, tile) coupling first measured on ``_foreach_maximum_List``
  reproduces here. Ragged lists keep the shipped masked 8192 body.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_list, _apply_scalar
from flag_gems.utils import libentry
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import (
    _TL_DTYPES,
    FALLBACK,
    MODE_LIST,
    MODE_SCALAR,
    MODE_SCALAR_LIST,
    OP_MIN,
    can_use_flat,
    fused_binary_scalar_kernel,
    run_binary,
)

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；均为编译期/启动期配置，不改变算子语义。
# 官方 do_bench kernel 口径（w1000/r100 median）kernel-only 变体扫描定档
# （Triton GCU 4.1.6，jobs/j20261119081737 fresh 重锚）：
#   * EXACT 免 mask 与 (dtype, tile) 耦合（maximum_List 判例在本栈复现）：
#     fp32 用 8192（-17/-11/-5% @256²/512²/1024²），16384 与之持平；
#     16-bit 用 16384（fp16 -5%/-4%，bf16 -4%），8192+EXACT 对 bf16 反回退
#     （+2~3%，512²/1024² 档）；
#   * 任一 tensor 元素数不整除所选 BLOCK 时整表回退 8192+mask（shipped
#     行为，逐位一致）；
#   * grid 恒为 tensor 数整除倍（_minimum_grid），nt=16 时与旧 _fused_grid
#     同为 48 program，其他列表长度修复尾部 tensor 漏写洞；
#   * num_warps 8 与 4 逐档相同（同 addcmul 家族判例），num_stages 后端不
#     流水化该 tile 循环，维持 1。
FIXED_GRID = 48
BLOCK_SIZE = 8192
BLOCK_SIZE_16BIT = 16384
NUM_WARPS = 4
NUM_STAGES = 1
USE_DEVICE_CONTEXT = 0
FP32_PROMOTE = 0
FUSED_LAUNCH = 1
FALLBACK_TO_PUBLIC = 0
# numel 全部整除所选 BLOCK 时走免 mask 的 EXACT kernel 变体（按 dtype 选
# BLOCK 后判定）；任何 tensor 不整除即整体回退 mask 路径。
USE_EXACT = 1
# 全部 overload 复用 _device_meta 只读缓存，消除稳态调用的同步 H2D。
USE_META_CACHE = 1
# The fp32 scalar table for .ScalarList is cached separately by (device, value).
USE_SCALAR_CACHE = 1
SCALAR_CACHE_LIMIT = 64
# --- TUNING PARAMS END ---

# Read-only metadata cache: the key contains only device and address/element count, not tensor data; the kernel still
# fully reads inputs and recomputes each time, so address reuse or in-place content changes do not affect correctness.
_SCALAR_CACHE = {}


@libentry()
@triton.jit
def _minimum_list_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
):
    """``.List`` minimum：一次 launch 处理整个配对 tensor 列表。

    ``meta_ptr`` 依次存放输入指针、输出指针、第二操作数指针、元素数
    （各 NT 项，int64）。grid.x 按 ``(tensor, part)`` 线性展开，且按构造
    为 NT 的整除倍（见 ``_minimum_grid``）。与公共 ``fused_binary_list_kernel``
    同构，差异有二：OP 固定为 minimum（免去编译期分派链），以及
    ``EXACT`` 在每个元素数都整除 ``BLOCK`` 时免去逐 tile 的 bounds
    mask——官方 do_bench kernel 口径下小 shape 档 mask 谓词是主要
    kernel 侧开销。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tensor_id = pid % NT
    part_id = pid // NT
    num_parts = tl.cdiv(num_programs, NT)

    a_raw = tl.load(meta_ptr + tensor_id)
    out_raw = tl.load(meta_ptr + NT + tensor_id)
    b_raw = tl.load(meta_ptr + 2 * NT + tensor_id)
    num_elems = tl.load(meta_ptr + 3 * NT + tensor_id)

    a_ptr = a_raw.to(tl.pointer_type(A_DT))
    out_ptr = out_raw.to(tl.pointer_type(A_DT))
    b_ptr = b_raw.to(tl.pointer_type(A_DT))

    arange = tl.arange(0, BLOCK)
    num_tiles = tl.cdiv(num_elems, BLOCK)
    for block_id in tl.range(part_id, num_tiles, num_parts):
        offs = block_id * BLOCK + arange
        if EXACT:
            a = tl.load(a_ptr + offs)
            b = tl.load(b_ptr + offs)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            tl.store(out_ptr + offs, tl.minimum(a, b))
        else:
            mask = offs < num_elems
            a = tl.load(a_ptr + offs, mask=mask)
            b = tl.load(b_ptr + offs, mask=mask)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            tl.store(out_ptr + offs, tl.minimum(a, b), mask=mask)


def _minimum_grid(self, fixed_grid, block_size):
    """grid 恒为 tensor 数整除倍：每个 tensor 分到相同的 part 数。

    ``(tensor, part)`` 线性化要求 ``num_programs % NT == 0``，否则尾部
    tensor 缺 part、按 ``num_parts`` 跨步的 tile 循环会跳过无人覆盖的块
    （``_foreach_flat._fused_grid`` 以 24 grid 承载 16-tensor 列表时实测
    半数张量奇数 tile 漏写，见 ``_foreach_maximum_List`` 的对拍记录）。
    这里按构造保证整除。
    """
    num_tensors = len(self)
    blocks_per_tensor = triton.cdiv(max(t.numel() for t in self), block_size)
    parts = max(1, min(fixed_grid // num_tensors, blocks_per_tensor))
    return (num_tensors * parts,)


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


def _run_minimum_cached(self, operand, mode, inplace):
    """三种操作数形态的精简单 launch 路径（metadata 走只读缓存）。

    不适用的形态返回 ``FALLBACK``，由调用方决定回退目标：广播 ``.List``
    需要 ``run_binary`` 的逐 tensor kernel；其余形态回退公共执行器。
    metadata 的四段布局（输入/输出/第二操作数地址 + 各元素数）与
    ``_foreach_flat`` 的融合 kernel 完全一致。

    配对 ``.List`` 走 ``_minimum_list_kernel``：BLOCK 按 dtype 分档
    （fp32 用 8192、16-bit 用 16384）后判定 ``EXACT`` 免 mask；任一
    tensor 的元素数不整除所选 BLOCK 时整表回退 8192 + mask，与历史
    shipped 行为一致。
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

    if mode == MODE_LIST:
        if not all(
            other.numel() == tensor.numel() for tensor, other in zip(self, operand)
        ):
            # Broadcast .List does not share flat indices with the inputs, so hand it to the per-tensor kernel.
            return FALLBACK
        b_ptrs = [other.data_ptr() for other in operand]
        meta = _device_meta(in_ptrs + out_ptrs + b_ptrs + numels, device)
        if USE_EXACT and dtype is not torch.float32:
            block = BLOCK_SIZE_16BIT
        else:
            block = BLOCK_SIZE
        if USE_EXACT and all(numel % block == 0 for numel in numels):
            exact = 1
        else:
            block = BLOCK_SIZE
            exact = 0
        grid = _minimum_grid(self, FIXED_GRID, block)
        _minimum_list_kernel[grid](
            meta,
            num_tensors,
            triton_dtype,
            FP32_PROMOTE,
            block,
            exact,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
    elif mode == MODE_SCALAR_LIST:
        if USE_SCALAR_CACHE:
            scalar_tensor = _cached_scalar_tensor(operand, device)
        else:
            scalar_tensor = torch.tensor(operand, dtype=torch.float32, device=device)
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        grid = _minimum_grid(self, FIXED_GRID, BLOCK_SIZE)
        fused_binary_scalar_kernel[grid](
            meta,
            scalar_tensor,
            num_tensors,
            OP_MIN,
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
    else:
        meta = _device_meta(in_ptrs + out_ptrs + [0] * num_tensors + numels, device)
        grid = _minimum_grid(self, FIXED_GRID, BLOCK_SIZE)
        fused_binary_scalar_kernel[grid](
            meta,
            self[0],
            num_tensors,
            OP_MIN,
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
        result = _run_minimum_cached(self, operand, mode, inplace)
        if result is not FALLBACK:
            return result
        if mode != MODE_LIST:
            # can_use_flat failed: leave dtype policy and exception semantics to the common executor.
            return fallback()
    result = run_binary(
        self,
        operand,
        OP_MIN,
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


def _foreach_minimum_List(self, other):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM_SCALARLIST")
    return _dispatch(
        self,
        other,
        MODE_LIST,
        False,
        lambda: _apply_list("minimum", self, other, False),
    )


def _foreach_minimum__List(self, other):
    return _dispatch(
        self,
        other,
        MODE_LIST,
        True,
        lambda: _apply_list("minimum", self, other, True),
    )


def _foreach_minimum_Scalar(self, scalar):
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        False,
        lambda: _apply_scalar("minimum", self, scalar, False),
    )


def _foreach_minimum__Scalar(self, scalar):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM__SCALAR")
    return _dispatch(
        self,
        scalar,
        MODE_SCALAR,
        True,
        lambda: _apply_scalar("minimum", self, scalar, True),
    )


def _foreach_minimum_ScalarList(self, scalars):
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        False,
        lambda: _apply_scalar("minimum", self, scalars, False),
    )


def _foreach_minimum__ScalarList(self, scalars):
    logger.debug("GEMS_GCU400 _FOREACH_MINIMUM__SCALARLIST")
    return _dispatch(
        self,
        scalars,
        MODE_SCALAR_LIST,
        True,
        lambda: _apply_scalar("minimum", self, scalars, True),
    )
