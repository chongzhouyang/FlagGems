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

NUM_SIPS = 24

# int32 offset safe domain: the masked tail block evaluates
# block_id * BLOCK + arange for masked lanes as well, reaching up to
# num_blocks * BLOCK - 1; keep n_total <= 2^31 - 2^17 (BLOCK <= 131072)
# so flat int32 offsets cannot wrap negative on the tail block.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)


@libentry()
@triton.jit(do_not_specialize=["n_elements"])
def _functional_sym_constrain_range_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)

    num_blocks = (n_elements + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < n_elements
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(y_ptr + off, x, mask=mask)


# GCU400 retune (2026-11-16, abs/rsqrt 1R1W family recipe, interleaved
# do_bench(median) A/B): the fixed single-tier B32768/nw4/ns4 launch with an
# int64 program id left the 16-bit big bucket 1.31x off the ~2.95TB/s DRAM
# copy floor and the tiny bucket on one oversized masked CTA. Reuse the
# abs-family size/itemsize dispatch: tiny wants the exact next_pow2 tile,
# 16-bit big wants B131072/nw2, fp32 stays on B65536/nw8 at the floor.
def _choose_launch(n_total, itemsize):
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 64), 4
    if n_total <= 65536:
        return 8192, 8
    if n_total <= 524288:
        return 32768, 8
    if itemsize == 4:
        return 65536, 8
    if n_total > (1 << 28):
        return 131072, 2
    return 65536, 2


def _launch_copy(src, dst):
    n_total = src.numel()
    if n_total == 0:
        return
    block, num_warps = _choose_launch(n_total, src.element_size())
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS * 2)
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic, host-level userland flake windows on shared
    # nodes); retrying the launch recompiles and succeeds (abs.py /
    # neg.py / rad2deg.py precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(src.device):
                _functional_sym_constrain_range_kernel[(grid_size,)](
                    src,
                    dst,
                    n_total,
                    BLOCK=block,
                    num_warps=num_warps,
                )
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "functional_sym_constrain_range copy kernel launch failed "
                "(attempt %d): %s",
                attempt + 1,
                exc,
            )


def _functional_sym_constrain_range(*args, **kwargs):
    logger.debug("GEMS_ENFLAME FUNCTIONAL_SYM_CONSTRAIN_RANGE")
    # Get the dep_token tensor to determine output device and properties
    tensor_arg = None
    for a in args:
        if isinstance(a, torch.Tensor):
            tensor_arg = a
            break
    if tensor_arg is None:
        for v in kwargs.values():
            if isinstance(v, torch.Tensor):
                tensor_arg = v
                break

    if tensor_arg is not None:
        # Return a copy of the dep_token tensor
        # The actual constraint validation is handled by PyTorch's symbolic tracing
        if tensor_arg.is_contiguous() and 0 < tensor_arg.numel() < _INT32_SAFE_LIMIT:
            output = torch.empty_like(tensor_arg)
            _launch_copy(tensor_arg, output)
            return output
        # Fallback for non-contiguous, empty or int32-unsafe tensors
        return tensor_arg.clone()

    # Fallback: return args[0] if it's a tensor
    return args[0] if len(args) > 0 else None
