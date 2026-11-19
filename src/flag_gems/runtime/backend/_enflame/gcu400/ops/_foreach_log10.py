# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_log10``（out-of-place）：委托公共 unary 执行器。

历史版本逐 tensor 一维 flat kernel 绕开公共执行器的 grid.y 硬件上限
（BLOCK 65536 / FIXED_GRID 48），但每条 tensor 一次 ``@libentry`` 分发
~11-15us host 成本，官方 bench core 下 16 tensor 列表延迟恒定
~240us 与 shape 无关，全档回退（256^2 0.31x、512^2 0.43x、1024^2
0.88x，2026-11-16 本机基线 mean 0.533x）。本目录 ``_foreach_unary.py``
的 lean 单 launch 路由（同 dtype 浮点连续列表 host ~8us，
``_device_meta`` 记忆化 + launch replay，见 2026-11-16 24 变体战役
记录）已按 2 的幂放大 BLOCK 把 grid.y 压回 255 以内，一次 launch 完成
全部张量；log10 行 fn 为 ``tl.log(x)*log10(e)`` 数学等价形态（log 调用
单价低 8-17%，``_foreach_log2.py`` 同款先例），out-of-place 直接委托。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary``
回落公共分组路径，dtype 提升（整型/布尔输出 float32）与异常语义保持
不变。
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_log10(self):
    logger.debug("GEMS_GCU400 _FOREACH_LOG10")
    return apply_unary("log10", self, inplace=False)
