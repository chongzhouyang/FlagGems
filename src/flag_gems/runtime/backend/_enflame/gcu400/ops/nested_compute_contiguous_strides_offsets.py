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

"""GCU400-specific ``nested_compute_contiguous_strides_offsets``.

This operator is pure host-side metadata arithmetic: input is an int64 ``(ntensors, dim)`` size matrix,
output is two small CPU metadata tensors, strides and offsets. The common implementation uses numpy's
vectorized reduction instead of ATen's C++ host loop; rewriting that path as a Triton device kernel
only adds one launch and host/device copies (KernelGen's Triton version measured
just 0.074x), so the host arithmetic main path is kept.

Optimization conclusion: among 13 strategies, the pure Python scalar path degraded to
0.53x on shapes like (1024, 3); the baseline "reverse view + cumprod + astype chain" had redundant intermediate arrays.
The final implementation writes strides in place with a single reverse recurrence and offsets with a single cumsum,
reducing gem-side total latency from 0.012637ms to 0.012552ms (3-round mean) and removing the ``np.errstate`` context
and one ``np.zeros`` allocation. Semantics are word-for-word aligned with the common implementation: 0-dim input returns
itself,
(N, 0) returns iota offsets, dtype / dimension errors, row products truncated to int32 before accumulation,
non-contiguous inputs read in raw data-pointer order.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ATen scalar type name, used in dtype error messages.
_DEBUG_TAG = "GEMS _NESTED_COMPUTE_CONTIGUOUS_STRIDES_OFFSETS"

# Use the scalar loop when the number of elements does not exceed this threshold (the measured crossover point with the
# numpy path, see
# ``_compute_strides_offsets_python``).
_SMALL_ELEMENTS = 16


_ATEN_TYPE_NAMES = {
    torch.bool: "Bool",
    torch.uint8: "Byte",
    torch.int8: "Char",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.float16: "Half",
    torch.float32: "Float",
    torch.float64: "Double",
    torch.bfloat16: "BFloat16",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
}


def _raw_matrix(sizes, ntensors, orig_dim):
    """Read int64 elements in raw data-pointer order, returning a numpy (N, D) matrix.

    Non-contiguous inputs must not use ``.contiguous()``: ATen interprets
    ``data_ptr()`` directly as a contiguous window, so ``as_strided`` reproduces the same window.
    """
    if sizes.is_contiguous():
        return sizes.cpu().numpy()
    flat = torch.as_strided(sizes, (ntensors * orig_dim,), (1,), sizes.storage_offset())
    return flat.cpu().numpy().reshape(ntensors, orig_dim)


def _compute_strides_offsets_python(mat):
    """小规模矩阵：标量循环优于多次 numpy ufunc 分派。

    每次 numpy ufunc 调用有 ~1-2us 的固定分派成本，元素数很少时（实测
    ``ntensors * orig_dim <= 16``）逐行标量展开更快；语义与 numpy 路径一致
    （行乘积按 ATen 的 int32 累加器语义截断后以 int64 累积）。

    Python 整数是无界的，而 ATen 在 int64 上回绕：所有 stride 先按 64 位掩码
    取补码，再用 uint64 视图转 int64，避免超出 int64 范围时溢出报错。
    """
    stride_rows = []
    offsets = np.empty(mat.shape[0], dtype=np.int64)
    accumulated = 0
    for row_index, row in enumerate(mat.tolist()):
        stride_row = [1] * len(row)
        product = 1
        for col_index in range(len(row) - 1, -1, -1):
            stride_row[col_index] = product & 0xFFFFFFFFFFFFFFFF
            product *= row[col_index]
        stride_rows.append(stride_row)
        offsets[row_index] = accumulated
        wrapped = product & 0xFFFFFFFF
        accumulated += wrapped - 0x100000000 if wrapped >= 0x80000000 else wrapped
    strides = np.array(stride_rows, dtype=np.uint64).view(np.int64)
    return strides, offsets


def _compute_strides_offsets(mat):
    """单次反向递推产出 strides，一次 cumsum 产出 offsets。

    ``strides[j] = prod(sizes[j+1:])`` 从右向左递推；整行乘积等于
    ``strides[0] * sizes[0]``，按 ATen 的 int 累加器语义截断到 int32 再
    符号扩展回 int64。所有 numpy 运算在 int64 补码下自然回绕，无需
    ``np.errstate`` 上下文。
    """
    ntensors, orig_dim = mat.shape
    if ntensors * orig_dim <= _SMALL_ELEMENTS:
        return _compute_strides_offsets_python(mat)
    # Only the rightmost column is always 1; all other columns are overwritten by the recurrence below, so use np.empty
    # and then
    # initialize a single column: full fill with np.ones measured 2.33us on (1024, 3), while
    # np.empty plus one column assignment takes only about 1.8us.
    strides = np.empty((ntensors, orig_dim), dtype=np.int64)
    strides[:, orig_dim - 1] = 1
    for col_index in range(orig_dim - 2, -1, -1):
        np.multiply(
            strides[:, col_index + 1],
            mat[:, col_index + 1],
            out=strides[:, col_index],
        )
    row_products = np.multiply(strides[:, 0], mat[:, 0])
    # ATen's row accumulator is ``int`` (int32): truncate the row product to int32 first, then
    # accumulate as int64 to reproduce the same semantics. ``np.add.accumulate``/``cumsum`` are the same ufunc
    # reduction, but accumulate measured ~2.3us faster than cumsum on current numpy ((1024, 3)
    # offsets: 6.55us -> 4.21us), so accumulate is used directly here.
    addends = row_products.astype(np.int32)
    offsets = np.empty(ntensors, np.int64)
    offsets[0] = 0
    np.add.accumulate(addends[:-1], dtype=np.int64, out=offsets[1:])
    return strides, offsets


def nested_compute_contiguous_strides_offsets(
    nested_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute contiguous strides and storage offsets from nested sizes (CPU metadata)."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(_DEBUG_TAG)

    sizes = nested_size

    # 0-dimensional input: ATen directly uses the input object itself as strides, without a dtype check.
    if sizes.dim() == 0:
        return sizes, torch.empty((0,), dtype=torch.int64, device="cpu")

    # ATen reads size(1) before checking dtype, so a 1-dimensional input raises a dimension error regardless of dtype.
    ntensors = sizes.size(0)
    orig_dim = sizes.size(1)

    if sizes.dtype != torch.int64:
        raise RuntimeError(
            "expected scalar type Long but found "
            + _ATEN_TYPE_NAMES.get(sizes.dtype, str(sizes.dtype))
        )

    # (N, 0): strides is still the input itself, offsets is iota.
    if orig_dim == 0:
        offsets = torch.arange(ntensors, dtype=torch.int64)
        return sizes, offsets

    # (0, D>0): ATen SIGSEGVs, so return a well-defined empty result here.
    if ntensors == 0:
        return torch.empty_like(sizes), torch.empty((0,), dtype=torch.int64)

    mat = _raw_matrix(sizes, ntensors, orig_dim)
    strides, offsets = _compute_strides_offsets(mat)

    return torch.from_numpy(strides), torch.from_numpy(offsets)
