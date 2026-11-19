# Copyright 2026, The FlagOS Contributors.
# GCU400-specialized true division: contiguous same-shape tensors go through a hand-written Triton kernel, everything
# else falls back to the generic implementation.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.div import true_divide_ as _generic_true_divide_
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _tdiv_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask, other=0)
    b = tl.load(b_ptr + offsets, mask=mask, other=1)
    tl.store(out_ptr + offsets, a / b, mask=mask)


def _launch(a, b, out, block, warps):
    n = out.numel()
    _tdiv_kernel[(triton.cdiv(n, block),)](a, b, out, n, BLOCK=block, num_warps=warps)
    return out


def _eligible(a, b):
    return (
        isinstance(a, torch.Tensor)
        and isinstance(b, torch.Tensor)
        and a.is_contiguous()
        and b.is_contiguous()
        and a.shape == b.shape
        and a.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def true_divide_tensor_(A, B):
    logger.debug("GEMS_ENFLAME TRUE_DIVIDE_")
    if _eligible(A, B):
        return _launch(A, B, A, 8192, 4)
    return _generic_true_divide_(A, B)
