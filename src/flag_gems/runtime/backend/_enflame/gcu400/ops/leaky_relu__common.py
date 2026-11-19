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
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


def _leaky_relu_autotune_configs():
    return [
        # Tiny tensors (n <= 32K): small blocks
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=8, num_stages=2),
        # Small-medium tensors (n ~ 64K-4M): 1024-element blocks
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),
        # Medium-large tensors (n ~ 4M-16M): 2048-element blocks
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=16, num_stages=5),
        # Large tensors (n >= 16M): 4096-element blocks for max bandwidth
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=5),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=5),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=5),
    ]


# restore_value: leaky_relu_ runs this kernel in-place (output_ptr aliases
# input_ptr). Autotuning benchmarks each config repeatedly on that buffer, so
# without restoring it the negative branch (x * negative_slope) compounds every
# trial and underflows to 0. Restoring output_ptr between trials keeps the input
# pristine; it is harmless for the out-of-place callers that pass a fresh buffer.
@libentry()
@triton.autotune(
    configs=_leaky_relu_autotune_configs(),
    key=["n_elements"],
    restore_value=["output_ptr"],
)
@triton.jit(do_not_specialize=["negative_slope"])
def _leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    output = tl.where(x >= 0, x, x * negative_slope)
    tl.store(output_ptr + offsets, output, mask=mask)


def leaky_relu(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU")
    if not A.is_contiguous():
        A = A.contiguous()
    output = torch.empty_like(A)
    n_elements = A.numel()
    if n_elements == 0:
        return output
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, output, n_elements, negative_slope)
    return output


@libentry()
@triton.jit(do_not_specialize=["n_elements", "negative_slope"])
def _leaky_relu_backward_kernel(
    grad_output_ptr,
    input_ptr,
    grad_input_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EVEN:
            grad_output = tl.load(grad_output_ptr + off)
            x = tl.load(input_ptr + off)
            grad_input = tl.where(x > 0, grad_output, grad_output * negative_slope)
            tl.store(grad_input_ptr + off, grad_input)
        else:
            mask = off < n_elements
            grad_output = tl.load(grad_output_ptr + off, mask=mask)
            x = tl.load(input_ptr + off, mask=mask)
            grad_input = tl.where(x > 0, grad_output, grad_output * negative_slope)
            tl.store(grad_input_ptr + off, grad_input, mask=mask)


_BWD_GRID_CAP = 48
_BWD_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# int32 safety domain: numel + BLOCK must stay < 2^31 (flat int32 offsets).
_BWD_MAX_NUMEL = (1 << 31) - (1 << 17)


def _leaky_relu_bwd_config(numel, dtype):
    # 2R1W ladder (tiers chosen by interleaved in-process do_bench A/B, GCU400/L600, 2026-11-16):
    #   16-bit: 2^24 65536/nw2 42.7/43.3µs, 2^29 1188/1110µs (32768/nw2 second best);
    #   fp32: 2^24 32768/nw2 75.4µs, 2^29 2358µs; native torch_gcu backward
    #   at the same shapes is on the order of 64ms/2359ms (reference implementation flaw; no comparison pressure).
    # Rescan on the new stack (Triton GCU 4.1.6 / flagtree d8ca7a9e, 2026-11-17, 5-9 rounds per arm in the same window,
    # interleaved do_bench, consistent across two independent sessions): the sweet spot for large tiers is unchanged
    # (16-bit >2^23 65536/nw2,
    #   fp32 >2^22 32768/nw2, grid cap 48 still optimal; 131072/nw4 is 5-15% worse across all tiers);
    #   fp32 mid band (4096,2^22] sweet-spot drift: 8192/nw4 wins over 16384/nw2 across the band (2^16
    #   4.14 vs 4.76µs, 2^20 10.09 vs 11.10, 2^21 14.40 vs 15.43, 2^22
    #   23.43 vs 23.99); from 2^23 it reverses back to the 32768/nw2 tier (41.77 vs 40.99).
    if dtype not in _BWD_DTYPES:
        if numel <= 1024:
            return 1024, 4
        if numel <= 32768:
            return triton.next_power_of_2(numel), 4
        return 8192, 4
    if numel <= 4096:
        block = triton.next_power_of_2(numel)
        return (256 if block < 256 else block), 4
    if dtype is torch.float32:
        if numel <= (1 << 22):
            return 8192, 4
        return 32768, 2
    if numel <= (1 << 21):
        return 8192, 4
    if numel <= (1 << 23):
        return 32768, 2
    return 65536, 2


def leaky_relu_backward(grad_output, self, negative_slope=0.01, self_is_result=False):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_BACKWARD")
    n_elements = self.numel()
    if n_elements >= _BWD_MAX_NUMEL:
        from flag_gems.ops.leaky_relu import leaky_relu_backward as _public

        return _public(grad_output, self, negative_slope, self_is_result)
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not self.is_contiguous():
        self = self.contiguous()
    grad_input = torch.empty_like(self)
    if n_elements == 0:
        return grad_input
    block_size, num_warps = _leaky_relu_bwd_config(n_elements, self.dtype)
    grid = min(triton.cdiv(n_elements, block_size), _BWD_GRID_CAP)
    even = n_elements % block_size == 0
    # aten leaky_relu_backward uses a strict `> 0` boundary: at x == 0 the
    # gradient is scaled by negative_slope (verified against torch.ops.aten).
    # self_is_result is ignored, matching the shared implementation above.
    # Cold-compile flake retry mirrors the forward kernel (square precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(self.device):
                _leaky_relu_backward_kernel[(grid,)](
                    grad_output,
                    self,
                    grad_input,
                    n_elements,
                    negative_slope,
                    BLOCK_SIZE=block_size,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return grad_input
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "leaky_relu backward flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def leaky_relu_(A, negative_slope=0.01):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_")
    if not A.is_contiguous():
        raise RuntimeError(
            "leaky_relu_ requires a contiguous tensor for in-place operation"
        )
    n_elements = A.numel()
    if n_elements == 0:
        return A
    # Autotuning re-runs the kernel on the launch arguments several times, so
    # launching with A as both source and destination would let an early
    # benchmark pass overwrite negative elements with scaled values that the
    # later passes then read, compounding the scaling. Compute into a scratch
    # buffer first and copy back, keeping the operation in-place from the
    # caller's perspective.
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        scratch = torch.empty_like(A)
        _leaky_relu_kernel[grid](A, scratch, n_elements, negative_slope)
        A.copy_(scratch)
    return A


def leaky_relu_out(A, negative_slope=0.01, *, out=None):
    logger.debug("GEMS_ENFLAME LEAKY_RELU_OUT")
    if out is None:
        return leaky_relu(A, negative_slope)
    if not A.is_contiguous():
        A = A.contiguous()
    n_elements = A.numel()
    if n_elements == 0:
        return out
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device.index):
        _leaky_relu_kernel[grid](A, out, n_elements, negative_slope)
    return out
