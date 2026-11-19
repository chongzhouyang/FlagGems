"""rrelu_with_noise_backward v2：分档 flat grid-stride + EXACT 免 mask 热循环。

v1（grid cap 48、固定 B4096/nw4、全程 masked）在 16-bit 大档仅跑出
1.81 TB/s（native 同窗 2.86 TB/s）。v2 按 special_xlogy/greater_/
copysign_ 已验证制式重标定（本 op 2026-11-16 两轮探针，do_bench median
w1000/r100，同窗含 v1 复刻锚点，与本文件 v1 数字吻合）：
  2^29 fp16  1784 -> 1091-1102 us（B65536/nw2，2.92-2.95 TB/s）
  2^29 bf16  1784 -> 1082-1083 us（B65536/nw2）
  2^29 fp32  2522 -> 2191-2212 us（B32768/nw2，2.91-2.94 TB/s）
  16.7M fp16 61.0 -> 40.9-41.1 us（B65536/nw2，超 xlogy 中带 B32768）
  16.7M fp32 79.7 -> 73.6-73.8 us（B65536/nw2）
  4096       3.4  -> 3.2-3.3 us（半幅 tile 双 CTA，launch 地板 ~2.3us）
- 16-bit 全程单档 B65536/nw2：2R1W 载入 tile 2x65536x2B=256KB，双
  program DSM 预算 448KB 线内，grid=48 真 48-way；fp32 大档 B32768/nw2
  （B65536 载入 tile 524KB 越线掉 24-way，copysign_ 制式）；fp32 中带
  (2^21, 2^25] B65536/nw2 实测 -1.2%@16.7M。
- EXACT constexpr（N%BLOCK==0 时热循环完全免 mask，两轮一致 -0~2.3%；
  16-bit B32768/nw4、fp32 B16384/nw4 等劣化档已探针否决）。
- TRAINING constexpr 分支保留：training=True 只载 GO/NZ（2R1W），
  不再像 TRACKED 版三载入两分支全算。
- P02 冷编译偶发失败重试 3 次（converterResult=-2 家族，xlogy 先例）。
数值路径与 v1/生产版一致（f32 中间量），精度门不回退。
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["rrelu_with_noise_backward"]

NUM_SLOTS = 48


@triton.jit(do_not_specialize=["N_total", "lower", "upper"])
def _rrwb_kernel(
    GO,
    X,
    NZ,
    OUT,
    N_total,
    lower,
    upper,
    TRAINING: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK_SIZE)
    num_blocks = (N_total + BLOCK_SIZE - 1) // BLOCK_SIZE
    slope = (lower + upper) * 0.5
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK_SIZE + arange
        if EXACT:
            go = tl.load(GO + off)
            go_f32 = go.to(tl.float32)
            if TRAINING:
                nz = tl.load(NZ + off)
                grad_f32 = go_f32 * nz.to(tl.float32)
            else:
                x = tl.load(X + off)
                grad_f32 = go_f32 * tl.where(x.to(tl.float32) > 0, 1.0, slope)
            tl.store(OUT + off, grad_f32.to(go.dtype))
        else:
            mask = off < N_total
            go = tl.load(GO + off, mask=mask, other=0)
            go_f32 = go.to(tl.float32)
            if TRAINING:
                nz = tl.load(NZ + off, mask=mask, other=0)
                grad_f32 = go_f32 * nz.to(tl.float32)
            else:
                x = tl.load(X + off, mask=mask, other=0)
                grad_f32 = go_f32 * tl.where(x.to(tl.float32) > 0, 1.0, slope)
            tl.store(OUT + off, grad_f32.to(go.dtype), mask=mask)


def _choose_block_and_warps(n, itemsize):
    # Small shapes hit the launch floor: half-width exact tile with dual CTA (xlogy tiny band convention).
    if n <= 4096:
        return max(triton.next_power_of_2(n) // 2, 256), 2
    if n <= 65536:
        return 2048, 4
    if n <= (1 << 21):
        return 8192, 4
    if itemsize == 2:
        # 16-bit single band: 2R1W with a 256KB load tile within the line at 48-way, best at both 16.7M/2^29 points.
        return 65536, 2
    if itemsize == 4 and n <= (1 << 25):
        # fp32 middle band measured -1.2%@16.7M; out of band, B65536 load tile is 524KB, exceeding the 448KB
        # dual-program line, dropping to 24-way; B32768/nw2 is the right choice for the large band (copysign_
        # convention).
        return 65536, 2
    return 131072 // itemsize, 2


def rrelu_with_noise_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    noise: torch.Tensor,
    lower: float,
    upper: float,
    training: bool,
    self_is_result: bool = False,
):
    # Explicit contiguous output: empty_like preserves non-contiguous strides,
    # while the kernel writes by linear offsets, so a non-contiguous out would scramble values.
    logger.debug("GEMS_ENFLAME RRELU_WITH_NOISE_BACKWARD")
    out = torch.empty(
        grad_output.shape, dtype=grad_output.dtype, device=grad_output.device
    )
    n = out.numel()
    if n == 0:
        return out
    go = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    x = input if input.is_contiguous() else input.contiguous()
    nz = noise if noise.is_contiguous() else noise.contiguous()
    block, num_warps = _choose_block_and_warps(n, out.element_size())
    grid = min(triton.cdiv(n, block), NUM_SLOTS)
    exact = n % block == 0
    # P02: the external llvm converter occasionally fails on cold compilation (converterResult=-2 family),
    # retrying the launch to trigger recompilation recovers (precedent: xlogy/add/sinh).
    for attempt in range(3):
        try:
            _rrwb_kernel[(grid,)](
                go,
                x,
                nz,
                out,
                n,
                float(lower),
                float(upper),
                TRAINING=bool(training),
                BLOCK_SIZE=block,
                EXACT=exact,
                num_warps=num_warps,
                num_stages=1,
            )
            return out
        except Exception:
            if attempt == 2:
                raise
    return out
