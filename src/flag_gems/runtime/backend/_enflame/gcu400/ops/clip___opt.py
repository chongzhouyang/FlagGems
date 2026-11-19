"""clip_ 候选 v2：复用 clip__opt 的 flat kernel 与 ladder（in-place 写回）。

生产 `clip_(A, mini, maxi)` 与 `clip()` 共用同一组分派对象，唯一差别是
`out0=A` 做 in-place 写回。本版沿用 clip__opt v2 的 flat grid-stride
kernel（2 流标量 bound，探针 2^30 fp16 in-place 1345us vs pointwise 3362us
vs native 1409us），eligible 时直接写回 A 本体；非连续 / 非浮点 /
tensor bound / launch 地板 / int32 不安全域 回落 pointwise（候选 v1 的
(1024,1,1) config 覆写对象）。host 分支保留 ValueError 语义。
"""

import logging

from flag_gems.runtime.backend._enflame.gcu400.ops.clip__opt import (
    _flat_eligible,
    _flat_plan,
    _launch_flat,
    clip_func,
    clip_func_max,
    clip_func_min,
)

logger = logging.getLogger(__name__)

__all__ = ["clip_"]


def clip_(A, mini=None, maxi=None):
    logger.debug("GEMS_ENFLAME CLIP_")
    if mini is None and maxi is None:
        raise ValueError("At least one of mini or maxi must not be None")
    if _flat_eligible(A, mini, maxi):
        plan = _flat_plan(A)
        if plan is not None:
            block, num_warps, body = plan
            _launch_flat(A, A, mini, maxi, A.numel(), block, num_warps, body)
            return A
    if mini is None:
        return clip_func_max(A, maxi, out0=A)
    if maxi is None:
        return clip_func_min(A, mini, out0=A)
    return clip_func(A, mini, maxi, out0=A)
