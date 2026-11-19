# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_log2_``（in-place）：委托公共 unary 执行器。

历史版本以逐 tensor 一维 kernel 绕开公共执行器的 grid.y 硬件上限
（256^2/512^2/1024^2 x 16 档位 grid.y 均不超限，但每条 tensor 一次
``@libentry`` 分发 ~11us host 成本），官方 bench core 下小中形状全面
回退（256^2 0.39x、512^2 0.53x，候选延迟恒定 ~175us 与 shape 无关）。
本目录 ``_foreach_unary.py`` 的 lean 单 launch 路由已按 2 的幂放大
BLOCK 把 grid.y 压回 255 以内，一次 launch 完成全部张量（host ~8us，
见 2026-11-16 24 变体战役记录），log2 行直接委托该执行器。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary``
回落公共分组路径，dtype 提升与异常语义（int64 in-place 抛
RuntimeError 等）保持不变。
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_log2_(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2_")
    return apply_unary("log2", self, inplace=True)
