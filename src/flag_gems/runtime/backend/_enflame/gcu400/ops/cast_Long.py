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
#
# GCU400 specialization of ``aten::_cast_Long`` (to int64).
#
# An int64 store itself is fine on GCU400 (an int64->int64 Triton copy is exact),
# but every *conversion* into int64 is not: the vector fp32->i64 convert, and the
# i32->i64 widening built on it, both write garbage (measured: 1.0 -> 0, -1.0 -> 1,
# 1000.5 -> -1). The generic FlagGems kernel relies on that convert, so on this
# backend every non-identity input produced wrong values, and float64 sources
# raised outright.
#
# Splitting the int64 output into two int32 words would be exact only for values
# inside the int32 range, and ``float -> int64`` is well defined outside it
# (2147483648.0 must stay 2147483648), so that shortcut would trade correctness
# for speed. The op is served by ATen's copy kernel through
# CompositeExplicitAutograd instead -- the same kernel the accuracy test's
# reference leg uses -- and adds no dispatch registration.

import logging

import torch

logger = logging.getLogger(__name__)

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def cast_Long(self, non_blocking=False):
    logger.debug("GEMS_ENFLAME CAST_LONG")
    if self.dtype == torch.int64:
        return self
    return torch.ops.aten._to_copy.default.redispatch(
        _FALLBACK_KEYSET,
        self,
        dtype=torch.int64,
        non_blocking=non_blocking,
    )
