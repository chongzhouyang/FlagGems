"""cosh 的 GCU400 专属 Triton 实现：out-of-place 1R1W flat grid-stride kernel。

替代公共 pointwise_dynamic codegen 路径（ENFLAME 默认配置 tile4096/block-ptr/
有限 CTA，512M fp16 档 2.25ms）。数学体按 dtype 分臂（探针 probe_r1/r2，
do_bench 中位 w1000/r100，新编译器 TopsRider 20261116，L600）：

- fp32 走 `0.5*(exp(x)+exp(-x))` 分组双 exp 体：512M 档 1396us 贴 copy 地板
  （1402us，两发 exp 被带宽完全掩盖）；分布式乘法形态 `exp(x)*0.5+exp(-x)*0.5`
  在 fp32 反劣 22%（1776us）——分组/分布怪癖与 16-bit 档方向相反（sinh 判例的
  镜像），IEEE 除 / rsqrt 替换单 exp 在 fp32 无收益（2134us 反劣，除法体
  fp32 灾难，sinh/cosh_ 家族同款否决）。
- fp16/bf16 走 `t + rsqrt(t)^2` 单 exp 体（t=exp(x)）：双 exp 体第二发超越
  函数在 16-bit 档无法被内存掩盖（512M: 分布式 1128us / 分组 1565us），恒等式
  cosh(x)=0.5*(t+1/t) 用硬件 rsqrt 原语平方替代倒数——常量分子 `/` 除法在
  分母为裸 exp 结果（非新鲜 add）时仍落慢路径（1287us，silu 快倒数判例不
  迁移），rsqrt(t)^2 体 512M: 928us、16.7M: 35.1us。±inf/NaN/中间溢出语义
  与双 exp 体逐位一致：exp 上溢 t=inf 时 rsqrt(inf)=0、平方=0，结果仍为
  inf；t 下溢为 0/subnormal 时 rsqrt 平方与 exp(-x) 同点溢出，cosh 大 |x|
  段两体同返 inf。

EVEN 免 mask 臂（N_total % BLOCK == 0 时裁掉边界谓词）：pow2 整除档
512M fp16 -2.6%、fp32 -4.2%；odd 尺寸 16-bit 走 masked+nw2（mask 谓词在
nw4 暴露 +12%），fp32 masked 保持 nw4。BLOCK/nw 阶梯按 2^12..2^29 边界
探针定标（probe_r2.txt）。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.cosh import cosh_func as _common_cosh_func
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Number of resident CTAs aligned with the GCU400 SIP topology; reused inside the kernel via a strided loop
NUM_SIPS = 48

# int32 offset safety range: masked tail blocks are also evaluated for masked lanes
# block_id*BLOCK+arange, with upper bound = num_blocks*BLOCK-1; when BLOCK<=131072,
# n_total <= 2^31-2^17 ensures the flat int32 offset does not wrap around to a negative value.
_INT32_SAFE_LIMIT = (1 << 31) - (1 << 17)

_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cosh_f32_flat_kernel(
    x_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK + arange
        if EVEN:
            value = tl.load(x_ptr + offsets).to(tl.float32)
            result = 0.5 * (tl.exp(value) + tl.exp(-value))
            tl.store(out_ptr + offsets, result)
        else:
            mask = offsets < N_total
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            result = 0.5 * (tl.exp(value) + tl.exp(-value))
            tl.store(out_ptr + offsets, result, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def _cosh_16bit_flat_kernel(
    x_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EVEN: tl.constexpr
):
    program_id = tl.program_id(0)
    num_programs = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(program_id, num_blocks, num_programs):
        offsets = block_id * BLOCK + arange
        if EVEN:
            value = tl.load(x_ptr + offsets).to(tl.float32)
            # cosh(x) = 0.5*(t + 1/t); 1/t is lowered to hardware primitives via the rsqrt(t)^2 identity,
            # eliminating the second transcendental evaluation and IEEE division slow paths in the double-exp body
            t = tl.exp(value)
            reciprocal_sqrt = tl.rsqrt(t)
            result = 0.5 * (t + reciprocal_sqrt * reciprocal_sqrt)
            tl.store(out_ptr + offsets, result)
        else:
            mask = offsets < N_total
            value = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            t = tl.exp(value)
            reciprocal_sqrt = tl.rsqrt(t)
            result = 0.5 * (t + reciprocal_sqrt * reciprocal_sqrt)
            tl.store(out_ptr + offsets, result, mask=mask)


def _choose_block(n_total, is_fp32):
    """小 shape 用 2 的幂 tile 多 CTA 摊 launch，大 shape 按 dtype 分档。"""
    if n_total <= 8192:
        return min(max(triton.next_power_of_2(n_total), 256), 2048), 1
    if is_fp32:
        # 2^16: 8192/nw4 (4.13 vs 4.67us @nw2); 2^18: 16384/nw4;
        # 2^21/2^22: 32768/nw4 (12.8/17.6us); 2^24: 65536/nw4 (51.2us);
        # 512M: 65536/nw4 (1396us at the floor; nw8 is equally fast and follows the family convention)
        if n_total <= (1 << 16):
            return 8192, 4
        if n_total <= (1 << 18):
            return 16384, 4
        if n_total <= (1 << 22):
            return 32768, 4
        return 65536, 4
    # 16-bit: 2^16: 4096/nw2 (4.07us); 2^18: 16384/nw4 (4.67us);
    # 2^21/2^22: 32768/nw4 (10.0/13.8-14.1us); 2^24: 65536/nw4 (35.1us);
    # 2^26: 65536/nw4 (114.8-116.2us); 512M: 131072/nw4 (official bench
    # 899-901/873-875us; nw2 is worse at 943-954us, and nw8 is -2.8% by probe measurement, but by the official
    # measurement
    # fp16 is unchanged and bf16 regresses by +4-6%; disproven and rolled back -- probe gains must be rechecked in
    # production form)
    if n_total <= (1 << 16):
        return 4096, 2
    if n_total <= (1 << 18):
        return 16384, 4
    if n_total <= (1 << 22):
        return 32768, 4
    if n_total <= (1 << 26):
        return 65536, 4
    return 131072, 4


def cosh(A):
    logger.debug("GEMS_ENFLAME COSH")
    # Non-floating-point / very large tensors use the common codegen path to preserve promotion and stride semantics
    if A.dtype not in _FLAT_DTYPES or A.numel() >= _INT32_SAFE_LIMIT:
        return _common_cosh_func(A)
    inp = A.contiguous()
    out = torch.empty_like(inp)
    n_total = inp.numel()
    if n_total == 0:
        return out

    is_fp32 = inp.dtype == torch.float32
    block, num_warps = _choose_block(n_total, is_fp32)
    even = n_total % block == 0
    if not even and not is_fp32:
        # Tail-block mask predicates show up at 16-bit nw4 (+12%); reduce odd sizes to nw2
        num_warps = 2
    kernel = _cosh_f32_flat_kernel if is_fp32 else _cosh_16bit_flat_kernel
    grid_size = min(triton.cdiv(n_total, block), NUM_SIPS)
    # Occasional cold-compilation failures in the external llvm converter (P02 toolchain with converterResult=-2
    # flake family); 3 launch retries ride out the transient window, and deterministic corruption still raises on the
    # 3rd attempt
    # (precedents in the same family: sinh.py / cosh___opt.py / atan2_.py)
    for attempt in range(3):
        try:
            with torch_device_fn.device(inp.device):
                kernel[(grid_size,)](
                    inp,
                    out,
                    n_total,
                    BLOCK=block,
                    EVEN=even,
                    num_warps=num_warps,
                )
            return out
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "cosh flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )


__all__ = ["cosh"]
