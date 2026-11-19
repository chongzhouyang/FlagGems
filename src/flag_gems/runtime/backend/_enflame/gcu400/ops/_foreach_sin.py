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
"""GCU400 专属 ``_foreach_sin``（out-of-place）：单 tensor flat 臂 + 委托执行器。

多 tensor 列表委托本目录 ``_foreach_unary.py`` 的 lean 单 launch 路由：历史
版本逐 tensor 一维 flat kernel 每条一次 ``@libentry`` 分发 ~10us host 成本，
官方 bench core 下候选延迟恒定 ~240-260us 与 shape 无关（256^2 仅
0.31-0.34x、9 cell 均值 0.679x，2026-11-18 新栈重锚实测；QA 周报 1114 的
0.28x 系旧栈陈旧值）。lean 路由按 2 的幂放大 BLOCK 把 grid.y 压回 255 以内，
一次 launch 完成全部张量（host ~8us，2026-11-16 24 变体战役；同表 ``sin_``
in-place 行同口径实测 3.93x）；表内 ``sin_func`` 的 fp32 提升语义与历史
kernel 的 ``.to(tl.float32)`` 分支逐位一致，零新数学风险。

单 tensor 列表保留 flat 专用臂：lean 路由的 BLOCK（16-bit 65536/131072）
在 len=1 档只给 1-8 个 CTA，list_length 轴实测 lean 11.6-20.9us，劣于老
flat kernel（BLOCK 2048 × grid<=48）的 4.7-12.2us（2026-11-18，256^2/512^2/
1024^2 x fp16/fp32 全档）；两臂各一次 launch，单 kernel 契约不破。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary``
回落公共分组路径，dtype 提升与异常语义（int64 in-place 抛 RuntimeError、
空列表/非 Tensor 元素对齐 ATen）保持不变。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# Best measured configuration for the single-tensor flat arm (same as the historical per-tensor implementation):
# BLOCK 2048 / grid limit 48 / num_warps 4, fastest across all tiers for small and medium single-tensor shapes.
_FLAT_BLOCK = 2048
_FLAT_FIXED_GRID = 48
_FLAT_NUM_WARPS = 4
_FLAT_NUM_STAGES = 1

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_sin_flat_kernel(
    in_ptr, out_ptr, N, BLOCK: tl.constexpr, FIXED_GRID: tl.constexpr
):
    """一维固定 grid 跨步执行逐元素 sin，fp32 提升后与执行器表体逐位一致。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(in_ptr + offs, mask=mask).to(tl.float32)
        result = tl.sin(value)
        tl.store(out_ptr + offs, result, mask=mask)


def _run_single_flat(tensor):
    out_tensor = torch.empty_like(tensor)
    num_elems = tensor.numel()
    grid = (min(triton.cdiv(num_elems, _FLAT_BLOCK), _FLAT_FIXED_GRID),)
    with torch_device_fn.device(tensor.device):
        _foreach_sin_flat_kernel[grid](
            tensor,
            out_tensor,
            num_elems,
            _FLAT_BLOCK,
            _FLAT_FIXED_GRID,
            num_warps=_FLAT_NUM_WARPS,
            num_stages=_FLAT_NUM_STAGES,
        )
    return out_tensor


def _foreach_sin(self):
    logger.debug("GEMS_GCU400 _FOREACH_SIN")
    if len(self) == 1:
        tensor = self[0]
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.dtype in _FLOAT_DTYPES
            and not tensor.dtype.is_complex
            and tensor.is_contiguous()
            and tensor.numel() > 0
        ):
            return [_run_single_flat(tensor)]
    return apply_unary("sin", self, inplace=False)
