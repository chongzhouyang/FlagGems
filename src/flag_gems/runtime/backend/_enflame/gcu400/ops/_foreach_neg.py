# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_neg``（out-of-place）：委托公共 unary 执行器。

历史版本用独立 int64 metadata 单 launch kernel 绕开公共执行器的 grid.y 硬件
上限，但 wrapper 每次调用都在 host 重建 metadata（3xNT 次 CPU 张量标量写 +
一次 pageable H2D，家族实测 ~136us，本机探针单件 177us）再走无 replay 缓存
的 Triton 分发，pageable H2D 与 stream 串行使 host 税泄漏进官方 bench 的
kernel 计时窗口：256^2/512^2/1024^2 x 16 全档 ~115-128us、与 shape 弱相关
（2026-11-16 本机基线 0.42-0.70x，baseline 档案 0.47-0.74x）。

本目录 ``_foreach_unary.py`` 的 lean 单 launch 路由（同 dtype 浮点连续列表
host ~8us，``_device_meta`` 按指针列表记忆化 + ``_launch_executor_kernel``
launch replay，见 2026-11-16 24 变体战役记录）已按 2 的幂放大 BLOCK 把
grid.y 压回 255 以内；neg 行 fn 为复用的单 tensor ``neg_func``（``-x``，
内存地板，无 extern），in-place 孪生 ``_foreach_neg_`` 同款先例（log2/
lgamma out-of-place 委托同族），out-of-place 直接委托该执行器。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary`` 回落
公共分组路径，dtype 策略（neg 保持输入 dtype、拒 bool）与异常语义不变
（neg 行在公共表与本目录副本中逐参一致）。
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_neg(self):
    logger.debug("GEMS_GCU400 FOREACH_NEG")
    return apply_unary("neg", self, inplace=False)
