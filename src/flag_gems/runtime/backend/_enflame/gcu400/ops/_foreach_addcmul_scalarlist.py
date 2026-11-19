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
"""GCU400 专属 ``_foreach_addcmul_.ScalarList`` 入口。

历史：本文件曾自带一套单 launch 融合 kernel（规避公共执行器二维
``grid=(block, tensor)`` 越过 GCU400 ``grid.y <= 255`` 上限的问题）。
``_foreach_binary`` flat 化之后，addcmul 全家族六个 overload 共用
``_apply_ternary`` 的 flat 单 launch 路径（int64 metadata 内容缓存、
一维 ``grid = NT * parts`` 恒整除倍、16-bit 走 fp32 opmath），本文件
收敛为薄委托：避免两套同构 kernel 各自漂移——11-16 flat 化在
``_flat_grid`` 引入的「grid 非 NT 整除倍漏 tile」缺陷正是双实现分叉
期间无人对照发现的产物。

dtype policy、异常语义与非连续/混合形态回退全部由 ``_apply_ternary``
统一承载：fast 路径服务同 dtype 连续浮点列表，其余形态落公共执行器。
"""

import logging

from ._foreach_binary import _apply_ternary

logger = logging.getLogger(__name__)


def _foreach_addcmul__ScalarList(self, tensor1, tensor2, value=1):
    """原地 ``self[i] += value[i] * tensor1[i] * tensor2[i]``。"""
    logger.debug("GEMS_GCU400 _FOREACH_ADDCMUL__SCALARLIST")
    return _apply_ternary("addcmul", self, tensor1, tensor2, value, True)
