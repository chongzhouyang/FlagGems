"""erf / erf_ 的 GCU400 专属实现。

连续浮点输入复用 special_erf 已验证的 fast_tanhf 快路径 kernel：deg-7 奇多项式
+ 硬件 fast_tanhf，全域 maxerr 1.35e-5（fp32 精度门 atol 1e-4 的 7 倍裕量）；
libdevice extern ``__ef_erff`` 为软件慢路径（512M 元素体耗时 ~5.2ms vs 拷贝地板
1.34ms），实测数字见 special_erf.erf_kernel docstring。erf_ 与 erf 共用同一
kernel：x_ptr/o_ptr 同址时单遍 load->store 逐元素无别名风险。
非浮点 / 非连续 / 超出 i32 安全域的输入回落 pointwise_dynamic codegen 路径，
保持 promotion 与 stride 语义。
"""

"""GCU400-specific implementation of erf / erf_.

Non-in-place erf keeps the backend's pointwise_dynamic codegen semantics; in-place erf_ uses a fixed number of
CTAs + grid-stride 1D flat kernel, removing codegen's Python-side dispatch and index decomposition overhead.
Input and output share one memory block, with one read and one write per element.
"""

import logging  # noqa: E402

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flag_gems.runtime import torch_device_fn  # noqa: E402

from ..utils.pointwise_dynamic import pointwise_dynamic  # noqa: E402
from .special_erf import erf_kernel  # noqa: E402

logger = logging.getLogger(__name__)

# SIP-resident CTA constant consistent with special_erf (reused by the strided loop inside the kernel)
NUM_SIPS = 24
# In the flat kernel, intra-block offsets are computed as i32, so N + BLOCK - 1 <= 2^31 - 1 must hold
_INT32_SAFE_NUMEL = 2**31 - 65536
_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def erf_func(x):
    return tl.math.erf(x.to(tl.float32))


def _choose_block(num_elements):
    """小 shape 用 1024 多 CTA 摊薄 launch（4096 元素档实测 -5~-16%，同 tan 定标）；
    中 shape 取 2 次幂；大 shape 65536（fast_tanhf 体为 DMA-bound，8192 劣化 +73%）。"""
    if num_elements <= 4096:
        return 1024
    if num_elements <= 32768:
        return triton.next_power_of_2(num_elements)
    return 65536


def _launch_fast(x, out):
    num_elements = x.numel()
    if num_elements == 0:
        return out
    block_size = _choose_block(num_elements)
    grid_size = min(triton.cdiv(num_elements, block_size), NUM_SIPS * 2)
    with torch_device_fn.device(x.device):
        erf_kernel[(grid_size,)](
            x,
            out,
            num_elements,
            BLOCK=block_size,
            HAS_TAIL=(num_elements % block_size) != 0,
            num_warps=2,
        )
    return out


def erf(x):
    logger.debug("GEMS_ENFLAME ERF")
    if x.dtype in _FAST_DTYPES and x.is_contiguous() and x.numel() <= _INT32_SAFE_NUMEL:
        return _launch_fast(x, torch.empty_like(x))
    return erf_func(x)


def erf_(A):
    logger.debug("GEMS_ENFLAME ERF_")
    if A.dtype in _FAST_DTYPES and A.is_contiguous() and A.numel() <= _INT32_SAFE_NUMEL:
        return _launch_fast(A, A)
    return erf_func(A, out0=A)


__all__ = ["erf", "erf_"]
