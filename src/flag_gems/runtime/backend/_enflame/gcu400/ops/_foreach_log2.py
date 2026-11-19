# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_log2``（out-of-place）：委托公共 unary 执行器。

历史版本用独立 int64 metadata 单 launch kernel 绕开公共执行器的 grid.y 硬件
上限，但 wrapper 每次调用都在 host 重建 metadata（3xNT 次 CPU 张量标量写 +
一次 pageable H2D，实测 ~136us）再走无 replay 缓存的 Triton 分发（~15us），
官方 bench kernel 模式下 256^2/512^2/1024^2 x 16 全档 ~105-213us、与 shape
弱相关（2026-11-16 本机基线 0.64-0.92x）。本目录 ``_foreach_unary.py`` 的
lean 单 launch 路由（同 dtype 浮点连续列表 host ~8us，``_device_meta`` 按
指针列表记忆化 + launch replay，见 2026-11-16 24 变体战役记录）已按 2 的幂
放大 BLOCK 把 grid.y 压回 255 以内，且 log2 行 fn 为 ``tl.log(x)*log2(e)``
数学等价变形（log 调用单价低 8-17%，``_foreach_log2_`` 同款先例），out-of-
place 直接委托该执行器。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary`` 回落
公共分组路径，dtype 提升（整型/布尔输出 float32）与异常语义保持不变。
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_log2(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2")
    return apply_unary("log2", self, inplace=False)
