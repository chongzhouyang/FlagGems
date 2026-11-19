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
"""GCU400 专属 ``_foreach_sign`` / ``_foreach_sign_``：len==1 flat 臂 + 委托执行器。

历史实现逐 tensor 发起一维 flat kernel：16 tensor 列表每次调用 16 条
``@libentry`` host 分发（~15us/条），官方 bench core 9 cell 候选延迟恒定
在 150-247us、与 shape/dtype 无关（2026-11-18 复测：out-of-place 均值
0.307x、in-place 均值 0.424x），1024^2 档约为 1R1W 内存地板的 10 倍。
多 tensor 列表改走本目录 ``_foreach_unary.py`` 的 lean 单 launch 路由
（公共 bulk 路径 host ~55us，lean ~8us；表内 ``_sign_func`` 的
``(x > 0) - (x < 0)`` 体与历史 kernel 的 fp32 提升三路 where 语义逐位
一致：NaN/+-0 -> 0，+-inf -> +-1），单 kernel 契约不破。

改后官方 bench core（kernel 模式，三轮 fresh）：out-of-place 9 cell 均值
0.307x -> 3.552/3.743x、in-place 0.424x -> 4.175/4.150x，绝对延迟
240/151us -> 8-63us 且随 shape 正常伸缩（256^2 贴 abs 同族地板 8-11us，
1024^2 fp16 48.1us 反超 abs 62.3us）；len>=2 与 len==1 两臂各一次
launch，单 kernel 契约不破。

单 tensor 列表保留 flat 专用臂：lean 路由按 dtype 放大 BLOCK（16-bit
131072 / 32-bit 65536），len==1 档只给 1-8 个 CTA，list_length 轴实测
劣于 flat kernel（BLOCK 2048 x grid<=48）4.7-12.2us 的量级（同表
``_foreach_sin`` 2026-11-18 判例）。

非浮点、非连续、mixed dtype/device 与 complex 输入由本地执行器回落
公共分组路径，dtype 提升与异常语义（空列表/非 Tensor 元素对齐 ATen）
保持不变。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Best measured configuration for the single-tensor flat arm (same setting as _foreach_sin on 2026-11-18:
# 256^2-1024^2 x fp16/fp32 len=1, all tiers 4.7-12.2us).
FLAT_BLOCK = 2048
FLAT_FIXED_GRID = 48
FLAT_NUM_WARPS = 4
FLAT_NUM_STAGES = 1
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_sign_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """一维固定 grid 跨步执行逐元素 sign（len==1 臂专用）。

    in_ptr 与 out_ptr 只在 out-of-place 变体下不同；fp32 提升后的三路
    where 与执行器表内 ``_sign_func`` 逐位一致（NaN 两侧比较均假 -> 0）。
    """
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.where(value > 0, 1.0, tl.where(value < 0, -1.0, 0.0))
        tl.store(out_ptr + offs, result, mask=mask)


def _single_tensor_ok(tensor):
    """len==1 flat 臂的进入条件：非空连续浮点 tensor。

    其余形态（非 Tensor 元素、整型/bool/complex、非连续、空张量）全部
    交给 ``apply_unary``，由其统一异常转换与分组回退对齐 ATen 语义。
    """
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.dtype in _FLOAT_DTYPES
        and not tensor.dtype.is_complex
        and tensor.is_contiguous()
        and tensor.numel() > 0
    )


def _run_single_flat(tensor, inplace):
    """单 tensor 一次 flat launch：in-place 直接写回，oop 分配新输出。"""
    out_tensor = tensor if inplace else torch.empty_like(tensor)
    num_elems = tensor.numel()
    grid = (min(triton.cdiv(num_elems, FLAT_BLOCK), FLAT_FIXED_GRID),)
    with torch_device_fn.device(tensor.device):
        _foreach_sign_flat_kernel[grid](
            tensor,
            out_tensor,
            num_elems,
            FLAT_BLOCK,
            FLAT_FIXED_GRID,
            num_warps=FLAT_NUM_WARPS,
            num_stages=FLAT_NUM_STAGES,
        )
    return None if inplace else [out_tensor]


def _foreach_sign(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIGN")
    if len(self) == 1 and _single_tensor_ok(self[0]):
        return _run_single_flat(self[0], inplace=False)
    return apply_unary("sign", self, inplace=False)


def _foreach_sign_(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIGN_")
    if len(self) == 1 and _single_tensor_ok(self[0]):
        return _run_single_flat(self[0], inplace=True)
    return apply_unary("sign", self, inplace=True)
