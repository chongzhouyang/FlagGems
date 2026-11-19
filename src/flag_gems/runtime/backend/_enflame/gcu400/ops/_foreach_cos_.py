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
"""GCU400 专属 ``_foreach_cos_``（in-place）：委托公共 unary 执行器。

历史版本以逐 tensor 一维 flat kernel 绕开公共执行器的 grid.y 硬件上限
（256^2/512^2/1024^2 x 16 档位 grid.y 均不超限，但每条 tensor 一次
``@libentry`` 分发 ~10us host 成本），官方 bench core 下候选延迟恒定
~172-224us 与 shape 无关（256^2 1.27-1.37x、512^2 2.3-2.5x、1024^2
5.5-6.8x，2026-11-16 实测复现）。本目录 ``_foreach_unary.py`` 的 lean 单
launch 路由已按 2 的幂放大 BLOCK 把 grid.y 压回 255 以内，一次 launch 完成
全部张量（host ~8us，见 2026-11-16 24 变体战役记录；同表 ``cos`` out-of-place
行同口径已实测 11.93x、``sin_`` in-place 行 13.05x），本行直接委托该执行
器；表内 ``cos_func`` 的 fp32 提升语义与历史 kernel 的 ``.to(tl.float32)``
分支一致，零新数学风险。

非浮点、非连续、mixed dtype/device 与 complex 输入由 ``apply_unary``
回落公共分组路径，dtype 提升与异常语义（int64 in-place 抛
RuntimeError 等）保持不变。
"""

import logging

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)


def _foreach_cos_(self):
    logger.debug("GEMS_GCU400 _FOREACH_COS_")
    return apply_unary("cos", self, inplace=True)
