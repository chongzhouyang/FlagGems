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
# Forward launch parameters (measured on GCU400): fixed 48 CTAs, 32K tile for large shapes.
_GRID_SIZE = 48
_TILE_BIG = 32768
_SMALL_N = 32768


@triton.jit
def _sigmoid(z):
    """后端快路径 sigmoid（tl.exp + tl.fdiv），比 libdevice tanh/erf 便宜一个量级。"""
    return tl.fdiv(1.0, 1.0 + tl.exp(-z))


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_kernel(
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """gelu 前向（含 inplace，Out 可与 X 同址）。

    erf 走 logistic 形式：Φ(x) = 0.5(1+erf(x/√2)) ≈ sigmoid(x·P(x²))，
    gelu = x·Φ(x)。P 由 minimax 拟合得到（fp32 4 阶，全实数域 |gelu 误差| ≤
    3.1e-6；16-bit 用 2 阶，误差 4.2e-5，远低于量化噪声）。相比 libdevice erff
    （约 24–30 个向量 op）本形式只需 11 个向量 op，且 tl.exp 走后端快 exp 通路。
    多项式**必须内联**写在循环体里：抽成 `@triton.jit` 子过程会让后端少做一层
    融合，实测同 shape 慢约 22%（1M fp32 0.0125→0.0151ms）。
    tanh 近似同样改用 sigmoid 形式：0.5x(1+tanh(u)) == x·sigmoid(2u)。
    """
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off).to(tl.float32)
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            t = x * x
            t2 = t * t
            if LOWPOLY:
                p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
            else:
                q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                p = (1.5956548825 + 0.0729405472 * t) + t2 * q
            y = x * _sigmoid(x * p)
        else:
            # c*(x + b*x^3) == (c + c*b*x^2)*x, saving one multiplication
            u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
            y = x * tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
        if EVEN:
            tl.store(Out + off, y)
        else:
            tl.store(Out + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_backward_kernel(
    DY,
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    inv_sqrt_2pi: tl.constexpr = 0.3989422804014327
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            dy = tl.load(DY + off).to(tl.float32)
            x = tl.load(X + off).to(tl.float32)
        else:
            mask = off < N
            dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if APPROX == 0:
            # d/dx [x·Φ(x)] = Φ(x) + x·φ(x); Φ uses the same logistic form as the forward pass
            t = x * x
            t2 = t * t
            if LOWPOLY:
                p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
            else:
                q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                p = (1.5956548825 + 0.0729405472 * t) + t2 * q
            dydx = tl.fdiv(1.0, 1.0 + tl.exp(-(x * p))) + x * (
                inv_sqrt_2pi * tl.exp(-0.5 * t)
            )
        else:
            u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
            s = tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
            # d/dx [x·sigmoid(2u)] = s + 2·x·s·(1-s)·u'
            dydx = s + 2.0 * x * s * (1.0 - s) * (
                0.7978845608028654 + 0.1070322243 * x * x
            )
        if EVEN:
            tl.store(Out + off, dydx * dy)
        else:
            tl.store(Out + off, dydx * dy, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _gelu_backward_kernel_h(
    DY,
    X,
    Out,
    N,
    APPROX: tl.constexpr,
    LOWPOLY: tl.constexpr,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """gelu_backward 半块分期版（fp32 大档专用）。

    外层 BLOCK tile 内部 static 两段半宽处理：fp32 下每段 128KB DMA 描述符
    （稳定粒度）+ 在飞 footprint 减半（393KB→低于满 tile 全量同时在飞），
    256M 元素实测 1486 vs 平铺 B32768 1733µs（-14.3%）。数学体与
    _gelu_backward_kernel 完全一致；EVEN 只按外层 BLOCK 判断，非整除时
    逐半块掩码。16-bit 勿用（实测反劣 28-70%）。
    """
    inv_sqrt_2pi: tl.constexpr = 0.3989422804014327
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    HALF: tl.constexpr = BLOCK // 2
    for block_id in tl.range(pid, num_blocks, num_pids):
        base = block_id * BLOCK
        for h in tl.static_range(2):
            off = base + h * HALF + tl.arange(0, HALF)
            if EVEN:
                dy = tl.load(DY + off).to(tl.float32)
                x = tl.load(X + off).to(tl.float32)
            else:
                mask = off < N
                dy = tl.load(DY + off, mask=mask, other=0.0).to(tl.float32)
                x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
            if APPROX == 0:
                # Same 4th-order Φ logistic form as the tiled version
                t = x * x
                t2 = t * t
                if LOWPOLY:
                    p = 1.59469875 + t * (0.0742564352 - 7.38255533e-4 * t)
                else:
                    q = (-2.512334e-4 - 6.08916e-5 * t) + t2 * 2.2226e-6
                    p = (1.5956548825 + 0.0729405472 * t) + t2 * q
                dydx = tl.fdiv(1.0, 1.0 + tl.exp(-(x * p))) + x * (
                    inv_sqrt_2pi * tl.exp(-0.5 * t)
                )
            else:
                u = (0.7978845608028654 + 0.035677408136300125 * x * x) * x
                s = tl.fdiv(1.0, 1.0 + tl.exp(-2.0 * u))
                dydx = s + 2.0 * x * s * (1.0 - s) * (
                    0.7978845608028654 + 0.1070322243 * x * x
                )
            if EVEN:
                tl.store(Out + off, dydx * dy)
            else:
                tl.store(Out + off, dydx * dy, mask=mask)


def _pick_tile(N):
    """按元素数选 (tile 宽度, CTA 数)。

    GCU400 实测：小 shape（≤32K 元素）用 next_pow2(N) 的精确 tile、单 CTA 一次
    迭代做完最省；更大的 shape 用 48 CTA 均分，tile 取 ceil(N/48) 的 2 次幂并
    封顶 32K。原实现对 fp32 直接给 65536，会让 64K 元素只剩 1 个 program
    （0.0206ms，比 torch 慢 4 倍），也是"大 tile 单 CTA"陷阱的来源。
    """
    if N <= _SMALL_N:
        return max(64, triton.next_power_of_2(N)), 1
    block = min(
        _TILE_BIG, max(2048, triton.next_power_of_2(triton.cdiv(N, _GRID_SIZE)))
    )
    return block, min(_GRID_SIZE, triton.cdiv(N, block))


# gelu_backward-specific tiers (measured 2026-11-18, using the official bench kernel convention w1000/r100,
# rechecked with a bench-style tensor layout):
# The 2R1W sweet spot differs from the forward 1R1W one; backward has its own tiers, do not mix with _pick_tile.
_BWD_TINY_2B = 4096
_BWD_MID1 = 1048576
_BWD_MID2 = 16777216
_BWD_STAGED = 16777216  # Starting point of the fp32 half-block staging
_BWD_STAGED_BLOCK = 65536


def _pick_tile_bwd(N, two_byte):
    """backward 专用 (kernel, tile 宽度, CTA 数, num_warps) 分档。

    GCU400 探针（官方 bench 同款 shape + bench 式张量布局复核）：
    - 2-byte tiny（≤4096）：B512/nw1 多 CTA 上 launch 地板（fp16 (64,64)
      3.40 vs 单 CTA 4.11µs）。
    - 2-byte (4096, 32K]：保持 np2 精确 tile 单 CTA（与 forward 同）。
    - 2-byte (32K, 1M]：B8192/nw2/g48（1M：10.6 vs 12.4µs）。
    - 2-byte (1M, 16M]：B16384/nw2（2^21..2^24：16.2/24.5/40.3/71.6µs 全档最优）。
    - 2-byte >16M：B32768/nw2（256M：976.5µs；B65536 反劣 17%；半块分期
      S8@131072 亦劣 28%，双 exp 数学链重时大 tile/分期均不受益）。
    - fp32 (32K, 16M)：B8192/nw4（2^16..2^23：4.9/7.7/10.0/15.4/23.2/37.7/66.3µs
      全档最优；64KB 粒度 tile 在飞 96KB 远离 DSM 线）。
    - fp32 ≥16M：半块分期 kernel（外层 65536、内部两段 32768；128KB 描述符）：
      16M：122.2 vs 125.7、256M：1486 vs 1733µs（-14.3%）。注意 fp32 平铺
      B16384（64KB 描述符）存在地址布局敏感病灶：bench 布局 142/2109µs vs
      干净布局 107.6/1550µs，勿用；B32768 平铺虽稳（126/1734）但被分期超越。
    一次调用直接给出 kernel/num_warps：bench kernel 模式的计时闭包含完整
    Python wrapper，二级条件链会在重负载下把 do_bench 256MB cache-clear
    (~128µs) 的遮蔽窗口顶穿（tiny 档实测 +0.7µs），分档须单帧收敛。
    """
    if two_byte:
        if N <= _BWD_TINY_2B:
            return _gelu_backward_kernel, 512, triton.cdiv(N, 512), 1
        if N <= _SMALL_N:
            return _gelu_backward_kernel, max(64, triton.next_power_of_2(N)), 1, 4
        if N <= _BWD_MID1:
            return _gelu_backward_kernel, 8192, min(_GRID_SIZE, triton.cdiv(N, 8192)), 2
        if N <= _BWD_MID2:
            return (
                _gelu_backward_kernel,
                16384,
                min(_GRID_SIZE, triton.cdiv(N, 16384)),
                2,
            )
        return (
            _gelu_backward_kernel,
            _TILE_BIG,
            min(_GRID_SIZE, triton.cdiv(N, _TILE_BIG)),
            2,
        )
    if N <= _SMALL_N:
        return _gelu_backward_kernel, max(64, triton.next_power_of_2(N)), 1, 4
    if N < _BWD_STAGED:
        return _gelu_backward_kernel, 8192, min(_GRID_SIZE, triton.cdiv(N, 8192)), 4
    return (
        _gelu_backward_kernel_h,
        _BWD_STAGED_BLOCK,
        min(_GRID_SIZE, triton.cdiv(N, _BWD_STAGED_BLOCK)),
        4,
    )


def _approx_code(approximate):
    return 1 if approximate == "tanh" else 0


def _launch_forward(inp, out, approximate):
    """前向 launch：分档 tile + 48 CTA 上限。

    小 shape 用精确 tile（单 CTA 一次迭代做完）；大 shape 用 ceil(N/48) 的 2 次幂
    tile（封顶 32K），使 CTA 数尽量铺满。2 字节 dtype 用 num_warps=2、fp32 用 4，
    掩码只在 tile 不能整除元素数时生成。
    """
    N = inp.numel()
    if N == 0:
        return out
    two_byte = inp.element_size() <= 2
    block, grid_size = _pick_tile(N)
    num_warps = 2 if (two_byte and N > _SMALL_N) else 4
    with torch_device_fn.device(inp.device):
        _gelu_kernel[(grid_size,)](
            inp,
            out,
            N,
            _approx_code(approximate),
            two_byte,
            (N % block) == 0,
            BLOCK=block,
            num_warps=num_warps,
        )
    return out


def gelu(self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU")
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(inp)
    return _launch_forward(inp, out, approximate)


def gelu_(A, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_")
    if not A.is_contiguous():
        A.copy_(gelu(A, approximate=approximate))
        return A
    return _launch_forward(A, A, approximate)


def gelu_backward(grad_output, self, *, approximate="none"):
    logger.debug("GEMS_ENFLAME GELU_BACKWARD")
    dy = grad_output if grad_output.is_contiguous() else grad_output.contiguous()
    inp = self if self.is_contiguous() else self.contiguous()
    out = torch.empty_like(dy)
    N = dy.numel()
    if N == 0:
        return out
    # 2R1W backward-specific tiers (see _pick_tile_bwd, a single call returns kernel + hyperparameters):
    # The 2-byte tiny tier with nw1 and more CTAs hits the launch floor; other 2-byte large tiers use nw2, and fp32
    # always uses nw4.
    kernel, block, grid_size, num_warps = _pick_tile_bwd(N, dy.element_size() <= 2)
    with torch_device_fn.device(dy.device):
        kernel[(grid_size,)](
            dy,
            inp,
            out,
            N,
            _approx_code(approximate),
            # Backward always uses 4th-order Φ (error 3.1e-6): the 16-bit 2nd-order arm's absolute error 4.2e-5 is
            # amplified by dy (~N(0,1)) and then combined with the tiny dydx≈0 values at x≈-4..-6, which in practice
            # causes
            # 0.1% of elements in test_gelu_backward to exceed the 1e-4 tolerance; the forward pass has no such
            # amplification but is still tiered.
            False,
            (N % block) == 0,
            BLOCK=block,
            num_warps=num_warps,
        )
    return out
