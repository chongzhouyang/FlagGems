# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``full``：手写 tl.full kernel 替代 pointwise_dynamic codegen。

生产 ``gcu400/ops/full.py`` 走 logger + check_dtype + torch.empty +
pointwise_dynamic codegen 生成的 full_func_scalar；纯填充算子的耗时几乎
全部在 codegen 的 Python 层。这里对 GCU 上的 fp16/fp32/bf16 标量填充
改走一次 Triton launch，其余 dtype / Tensor fill_value / 非 GCU 设备
一律委托生产实现，保证 dtype policy 与溢出检查语义不变。

launch 形态（对齐同族 new_full 三轮 RSI 定标，2026-11-16 GCU400 实测）：
- grid 封顶 24（sip_count=24，写路径 24 CTA 已饱和；grid 48 -3.7%、
  每块连续大 span 的 flat 16384-CTA 形态 fp16/bf16 仅 2406GB/s）；
- BLOCK 分档：n<=8192 单 CTA 精确块免 mask；更大取
  min(8192, max(2048, np2(ceil(n/48))))（B2048-B8192 全带持平最优，
  B32768 在 16K-64K 元素档劣 20-64%、2^20 fp32 劣 14%，单块 np2(n)
  >=2^20 元素触发 DSM OOM），grid-stride 循环核内跨步复用 program；
- num_warps=2（nw4 大档 -2%）；大档 2^30 全 BLOCK 钉写墙
  3246-3262GB/s（DRAM 写墙 3.25/3.26TB/s），中带 2^24 2581-2900GB/s。

pin_memory 守卫取 truthy（True 才委托生产）：ATen 注册链把
``pin_memory=False``（非 None）传给本函数，严格 ``is not None`` 判断会让
GCU fast path 在 torch.full/ATen dispatch 下永远走不到，退化到生产
codegen 版（topsprof 实证 full_func_scalar 702us = QA 口径 1.27x 的
真实执行体）；False 与 None 语义等价（torch.empty 均不 pin），仅
pin_memory=True 时委托生产实现。
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime.backend._enflame.gcu400.ops.full import full as _public
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["full"]

logger = logging.getLogger(__name__)

_FLAT_MAX_GRID = 24
_FLAT_CAP_BLOCK = 8192
_FLAT_MIN_BLOCK = 2048
_TINY_MAX_ELEMS = 8192
_FLAT_MAX_ELEMS = 2**31 - 1  # int32 offsets guard, larger falls back to _public

_FINFO = {
    torch.float16: torch.finfo(torch.float16),
    torch.float32: torch.finfo(torch.float32),
    torch.bfloat16: torch.finfo(torch.bfloat16),
}


@libentry()
@triton.jit
def _full_kernel(
    OUT,
    N,
    VAL,
    BLOCK_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """Each program produces a constant vector; with FIXED_GRID>0 programs are reused by in-kernel striding."""
    arange = tl.arange(0, BLOCK_SIZE)
    values = tl.full([BLOCK_SIZE], VAL, OUT.dtype.element_ty)
    if FIXED_GRID > 0:
        pid = tl.program_id(0)
        num_programs = tl.num_programs(0)
        for block_id in tl.range(pid, tl.cdiv(N, BLOCK_SIZE), num_programs):
            offs = block_id * BLOCK_SIZE + arange
            if MASKED:
                tl.store(OUT + offs, values, mask=offs < N)
            else:
                tl.store(OUT + offs, values)
    else:
        offs = tl.program_id(0) * BLOCK_SIZE + arange
        if MASKED:
            tl.store(OUT + offs, values, mask=offs < N)
        else:
            tl.store(OUT + offs, values)


def full(size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None):
    logger.debug("GEMS_ENFLAME FULL")
    dev = torch.device(device) if device is not None else torch.device("cpu")
    if dtype is None:
        if isinstance(fill_value, bool):
            dt = torch.bool
        elif isinstance(fill_value, int):
            dt = torch.int64
        else:
            dt = torch.get_default_dtype()
    else:
        dt = dtype

    if (
        dev.type != "gcu"
        or dt not in _FINFO
        or isinstance(fill_value, torch.Tensor)
        or layout is not None
        or (pin_memory is not None and pin_memory)
    ):
        return _public(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )

    # Same semantics as production check_dtype: bool scalars under a floating dtype are cast to int first, then
    # overflow-checked.
    if isinstance(fill_value, bool):
        fill_value = int(fill_value)
    if not (math.isinf(fill_value) or math.isnan(fill_value)):
        finfo = _FINFO[dt]
        if fill_value < finfo.min or fill_value > finfo.max:
            raise RuntimeError(
                f"value cannot be converted to type {dt} without overflow"
            )

    out = torch.empty(size, device=dev, dtype=dt)
    num_elems = out.numel()
    if num_elems == 0:
        return out
    if num_elems > _FLAT_MAX_ELEMS:
        # int32 offsets cannot address larger fills; keep semantics exact.
        return _public(
            size,
            fill_value,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
        )
    if num_elems <= _TINY_MAX_ELEMS:
        block = max(2, triton.next_power_of_2(num_elems))
    else:
        block = min(
            _FLAT_CAP_BLOCK,
            max(
                _FLAT_MIN_BLOCK,
                triton.next_power_of_2(-(-num_elems // _FLAT_MAX_GRID)),
            ),
        )
    grid = (min(triton.cdiv(num_elems, block), _FLAT_MAX_GRID),)
    _full_kernel[grid](
        out,
        num_elems,
        fill_value,
        BLOCK_SIZE=block,
        MASKED=(num_elems % block != 0) or (block > num_elems),
        FIXED_GRID=_FLAT_MAX_GRID,
        num_warps=2,
        num_stages=1,
    )
    return out
