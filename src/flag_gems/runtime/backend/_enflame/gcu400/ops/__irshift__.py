# Copyright 2026, The FlagOS Contributors.
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
"""Triton implementation of ``aten::__irshift__`` (in-place right shift).

``__irshift__.Tensor`` / ``__irshift__.Scalar`` 与 ``bitwise_right_shift_`` 是同一
就地右移语义的两个 ATen 入口（经算子 schema 与数值对拍确认）。这里直接复用
enflame 本地 ``bitwise_right_shift_`` 的分层实现：连续存储走单次 1-D flat launch
（大工作集用 streaming tile、小工作集用并行阶梯），非连续 / 广播 / 混合位宽退回
stride-aware pointwise 路径。

此前本文件自带一份默认配置的 ``pointwise_dynamic`` kernel，每次调用都要付
rank 特化 block-pointer launch 与完整的 promotion / stride 开销，大张量（≥16 MB
工作集，带宽bound）也只能拿到 pointwise 的 tile 效率；复用 flat 路径后同一份
launch 逻辑同时覆盖 tensor 与 scalar 右移，无需在本文件重复维护 kernel。
"""

import logging

import torch

from .bitwise_right_shift import bitwise_right_shift_

logger = logging.getLogger(__name__)


def __irshift__(self: torch.Tensor, other) -> torch.Tensor:
    """就地右移 ``self >>= other``，返回被就地修改的 ``self``。"""
    logger.debug("GEMS_ENFLAME IRSHIFT")

    # In-place semantics: out is fixed to self, and ATen's __irshift__ accepts only this form.
    bitwise_right_shift_(self, other)
    return self
