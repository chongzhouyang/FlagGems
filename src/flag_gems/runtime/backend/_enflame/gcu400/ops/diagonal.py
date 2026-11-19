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

import logging

import torch
import triton

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)


@pointwise_dynamic(is_tensor=[True], promotion_methods=[(0, "DEFAULT")])
@triton.jit
def copy_func(x):
    return x


def diagonal_backward(grad_output, input_sizes, offset, dim1, dim2):
    logger.debug("GEMS_ENFLAME DIAGONAL_BACKWARD")
    grad_input = torch.zeros(
        input_sizes, dtype=grad_output.dtype, device=grad_output.device
    )
    diag = torch.diagonal(grad_input, offset, dim1, dim2)
    # The pointwise kernel cannot write into the strided diagonal view directly:
    # for 4-D inputs whose diagonal dims are (0, 1) / (1, 0) every dtype and
    # offset came back wrong (the elementwise store lands on packed offsets
    # instead of the view strides). Copy through a contiguous staging buffer and
    # let the strided scatter follow the view metadata.
    if grad_output.numel() == 0:
        return grad_input
    staged = torch.empty(
        grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
    )
    copy_func.instantiate(grad_output.ndim)(grad_output, out0=staged)
    diag.copy_(staged)
    return grad_input
