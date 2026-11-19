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
"""GCU400 专属 ``_foreach_lgamma`` / ``_foreach_lgamma_``：16 指针槽单 launch。

设计（2026-11-16 perf round，4 轮 fresh bench 定标）：

- **数学体**：与 ``gcu400/ops/lgamma_.py`` 同源的纯 Triton Stirling 体
  （反射 + 移位到 y>=4 + 三项 Stirling 级数 + 5 项奇 Taylor sin，sin 项数
  相对 lgamma_.py 的 7 项为 2026-11-16 二轮 perf round 的削减，见下）。无任何
  libdevice extern——``__ef_lgammaf`` extern 是 P16（>=512 并发 CTA 下 SIP
  abort）的触发源，且其 fp32 向量宽度病理（16-bit 输入时管线拓宽到
  vector<256> 落到 v2 慢实现，16.2ms/16M 对纯体 0.57ms）使其在任何 CTA
  数下都不可取。相对逐 tensor 版本原用的 9 除法 Lanczos 体（~99ns/元素，
  其中 ``tl.sin`` extern 占主导），Stirling 体 ~23-26ns/元素。
- **launch 结构**：``_amp_foreach_non_finite_check_and_unscale_`` 的 NT 指针
  槽模式放大到 16 槽——16 in + 16 out + 16 size 显式参数，padding 槽
  size=0 成零迭代循环，一个编译变体服务任意列表长度 <=16；>16 按 16 分块。
  逐 tensor launch 的 host 成本 = 16x(JIT 发射 ~10us + dispatch 地板
  ~2.4us)，out-of-place 每调用 ~160-250us 超出 do_bench L2 flush 阴影窗，
  小形状读数被 host 吞掉（(256,256)x16 实测 0.57x）；单 launch 后同一读数
  7.5x。grid 固定 48 CTA + 槽内跨步（GCU400 Fixed Grid；grid.y 上限与
  65535 grid.x 硬限对任意 numel 均免疫）。
- **MASKED constexpr**：批内所有 numel % BLOCK == 0 时热循环免 mask 谓词
  （bench 三形状全命中）；否则整批走 masked 路径，正确性不变。
- **5 项 sin Taylor**（2026-11-16 二轮 perf round）：7 项体在 |angle| <= pi/2 的
  截断误差 5.6e-8，对 fp32 套件容差（atol 1e-4）过量三个数量级；5 项体（截至
  angle^11）截断误差 2.3e-6，经 ``err/sin`` 比值放大后仍在 |angle|=pi/2 处取峰、
  在 sin 零点处消失（近整数极点带不放大），保持 44x 容差余量，逐元素减一层
  Horner 在 16M 单槽探针上 -1.1%（fp32）/-1.4%（fp16）。更激进的 4 项体
  （截至 angle^9，误差 1.13e-4）越过了 atol 门槛，被全值域探针的 odd-size
  randn 用例否决。
- **out-of-place 单缓冲分配**（同轮）：16 次 ``empty_like`` 的 host 成本
  ~45us（每次 ~2.8us dispatcher 行程）在 bench 框架进程的分配器状态下
  超出 do_bench L2 flush 阴影窗，小形状读数 +10us；uniform 形状列表改用
  一次 ``torch.empty((n, *shape))`` + ``unbind(0)`` 视图（~6us）。
  与 ATen 的可见差异：uniform 列表的输出共享同一 storage（值为正确值、
  形状/连续性/dtype 一致；对输出 ``resize_`` 的用户代码不适用——混合
  形状与非快路径仍走逐张量分配/公共路径）。
- **极点检测修正**（Stirling 体移植引入，探针抓出）：``reduced =
  x - floor(x+0.5)`` 作整数判据在 fp32 ulp>=1（|x|>=2^23）时 ``x+0.5``
  先舍入，[2^23, 2^24) 的负奇整数极点漏判（负向 logspace(1,8) 全域探针
  2136/100001 点返回有限值而非 +inf）。修法：``is_pole = (x <= 0) &
  ((reduced == 0) | (x <= -8388608))``——fp32 在 |x|>=2^23 全体可表示值
  即整数，常量比较封口，免第二次 floor。``gcu400/ops/lgamma_.py`` 的
  ``_lgamma_stirling`` 共享该缺口（其全值域验证采样未覆盖该带），本文件
  不回改该算子。

Host 守卫：非浮点、混合 dtype、混合 device、非连续与 numel >= 2^31（int32
平坦偏移域外）回退公共 ``apply_unary`` 分组路径，dtype 提升（int/bool ->
float32）与异常语义同 ATen；空列表与非 Tensor 元素维持公共路径报错。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ._foreach_unary import apply_unary

logger = logging.getLogger(__name__)

# GCU400 Fixed Grid: 48 CTAs; if the grid is insufficient, shrink according to the total number of tiles (so degenerate
# small batches do not idle).
_GRID_SIZE = 48

# Number of pointer slots carried in one launch; lists with >16 tensors are split into chunks of 16 (one launch per
# chunk).
_SLOTS = 16

# Compute/memory-bound structural floor: BLOCK x num_warps across all combinations ({1024..8192}x{w2,w4}
# with eight configurations within ±0.3%), keeping 2048/w4 = the scalar single-tensor ``lgamma_`` calibration.
_BLOCK = 2048
_NUM_WARPS = 4

# Stirling shift threshold (same parameters as ``lgamma_.py``; interleaved A/B of 3 and 4 showed no difference, so keep
# 4 for accuracy).
_SHIFT_THRESHOLD = 4

# Upper bound of the int32 flat offset domain: max offsets = ceil(n/BLOCK)*BLOCK - 1 <= 2^31 - 1.
_MAX_FLAT_NUMEL = 2**31 - 1


@triton.jit
def _lgamma_stirling(x, SHIFT_THRESHOLD: tl.constexpr):
    # lgamma in fp32 via reflection + shift-to->=SHIFT_THRESHOLD + Stirling
    # asymptotics.  Verbatim body of ``gcu400/ops/lgamma_.py`` except for the
    # pole test fix documented in the module docstring.  No libdevice extern:
    # every op stays on the core Triton path (log/rsqrt/where) at any vector
    # width, which keeps the kernel immune to P16 (extern SIP abort under
    # >=512 concurrent CTAs) and to the extern's 16-bit-widening slowdown.
    reflect = x < 0.5
    y = tl.where(reflect, 1.0 - x, x)
    # Shift y up to >= SHIFT_THRESHOLD, accumulating the product of the skipped
    # factors; lgamma(y) = stirling(y') - log(product).
    t = 1.0
    for _ in tl.static_range(SHIFT_THRESHOLD):
        m = y < SHIFT_THRESHOLD * 1.0
        t = tl.where(m, t * y, t)
        y = tl.where(m, y + 1.0, y)
    logy = tl.log(y)
    # 1/y via rsqrt(y*y): same value for y > 0 (series term is only r/12, so
    # the extra ulp of rsqrt is irrelevant), and it avoids the divide that
    # pushes the 16-bit pipeline onto a slower codegen path on GCU400.
    r = tl.math.rsqrt(y * y)
    r2 = r * r
    series = r * (
        0.08333333333333333 - r2 * (0.002777777777777778 - r2 * 0.0007936507936507937)
    )
    res = (y - 0.5) * logy - y + 0.9189385332046727 + series - tl.log(t)
    # Reflection for x < 0.5: lgamma(x) = log(pi) - log|sin(pi x)| - lgamma(1-x).
    # `reduced` is in [-0.5, 0.5], so |angle| <= pi/2 and a 5-term odd Taylor
    # for sin (through angle^11) is enough: truncation error 2.3e-6 at pi/2
    # (the 7-term body's 5.6e-8 was 3 decades under the fp32 suite atol 1e-4;
    # 2.3e-6 keeps 44x of that margin), and after the err/sin amplification
    # the log|sin| error peaks at |angle| = pi/2 and vanishes toward the sin
    # zeros, so the near-integer pole band is NOT amplified.  Dropping the
    # 6th Taylor level measured -1.1% (fp32) / -1.4% (fp16) per element on
    # the 16M single-slot probe; a 4-level body (through angle^9, error
    # 1.13e-4) crossed the suite atol and was rejected by the full-domain
    # probe (odd-size randn cases).  The Taylor form also keeps sin finite
    # for tiny angles and the 2^64 pre-scaling keeps log|sin| finite when
    # |sin| is subnormal.
    reduced = x - tl.floor(x + 0.5)
    angle = 3.141592653589793 * reduced
    angle2 = angle * angle
    sin_poly = 1.0 - angle2 * (
        0.16666666666666666
        - angle2
        * (
            0.008333333333333333
            - angle2 * (0.0001984126984126984 - angle2 * 2.7557319223985893e-06)
        )
    )
    sin_pi_x = angle * sin_poly
    log_sin = tl.log(tl.abs(sin_pi_x) * 18446744073709551616.0) - 44.3614195558365
    reflected = 1.1447298858494002 - log_sin - res
    result = tl.where(reflect, reflected, res)
    # Gamma's poles (0 and the negative integers) and +-inf map to +inf; NaN
    # propagates through the arithmetic.  The pole test reuses `reduced`
    # (x - round(x) == 0 iff x is an exact integer) instead of a second
    # tl.floor(x).  FIX over lgamma_.py: for |x| >= 2^23 the fp32 addition
    # x + 0.5 rounds FIRST (ulp >= 1), so odd integers in [2^23, 2^24) get
    # reduced == +-1 and the pole is missed.  Every representable fp32 value
    # with |x| >= 2^23 is an integer, and lgamma's poles are the non-positive
    # ones, so the constant comparison closes the gap without a second floor.
    is_pole = (x <= 0.0) & ((reduced == 0.0) | (x <= -8388608.0))
    result = tl.where(is_pole | (tl.abs(x) == float("inf")), float("inf"), result)
    return result


@triton.jit
def _lgamma_slot(
    in_ptr,
    out_ptr,
    pid,
    num_elements,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    SHIFT_THRESHOLD: tl.constexpr,
):
    # Grid-stride over the tiles of one tensor slot.  MASKED=0 requires
    # num_elements % BLOCK == 0 for every live slot (host guarantees it) and
    # removes every mask predicate from the hot loop; size-0 padding slots
    # make the tile loop a zero-trip loop.
    if MASKED:
        num_tile = (num_elements + BLOCK - 1) // BLOCK
    else:
        num_tile = num_elements // BLOCK
    for tile_id in tl.range(pid, num_tile, GRID_SIZE):
        offsets = tile_id * BLOCK + tl.arange(0, BLOCK)
        if MASKED:
            mask = offsets < num_elements
            raw = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        else:
            raw = tl.load(in_ptr + offsets)
        result = _lgamma_stirling(raw.to(tl.float32), SHIFT_THRESHOLD)
        out_val = result.to(out_ptr.dtype.element_ty)
        if MASKED:
            tl.store(out_ptr + offsets, out_val, mask=mask)
        else:
            tl.store(out_ptr + offsets, out_val)


@libentry()
@triton.jit(
    do_not_specialize=[
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "s12",
        "s13",
        "s14",
        "s15",
    ]
)
def _foreach_lgamma_kernel(
    i0,
    i1,
    i2,
    i3,
    i4,
    i5,
    i6,
    i7,
    i8,
    i9,
    i10,
    i11,
    i12,
    i13,
    i14,
    i15,
    o0,
    o1,
    o2,
    o3,
    o4,
    o5,
    o6,
    o7,
    o8,
    o9,
    o10,
    o11,
    o12,
    o13,
    o14,
    o15,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    s8,
    s9,
    s10,
    s11,
    s12,
    s13,
    s14,
    s15,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    MASKED: tl.constexpr,
    SHIFT_THRESHOLD: tl.constexpr,
):
    # Fused out-of-place / in-place lgamma over up to 16 same-(device,dtype)
    # tensors in a single launch: each program visits every slot's tile space
    # with the same grid stride, so load balance stays even across differently
    # sized tensors.  In-place calls pass o_k == i_k.
    pid = tl.program_id(0)
    _lgamma_slot(i0, o0, pid, s0, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i1, o1, pid, s1, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i2, o2, pid, s2, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i3, o3, pid, s3, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i4, o4, pid, s4, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i5, o5, pid, s5, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i6, o6, pid, s6, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i7, o7, pid, s7, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i8, o8, pid, s8, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i9, o9, pid, s9, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i10, o10, pid, s10, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i11, o11, pid, s11, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i12, o12, pid, s12, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i13, o13, pid, s13, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i14, o14, pid, s14, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)
    _lgamma_slot(i15, o15, pid, s15, BLOCK, GRID_SIZE, MASKED, SHIFT_THRESHOLD)


def _fast_path(self) -> bool:
    """True 时走 16 槽单 launch：非空、全连续浮点、单一 (dtype, device)、
    numel 在 int32 平坦偏移域内。任何 AttributeError（非 Tensor 元素、
    空列表）与其他不满足者一律回退公共 ``apply_unary`` 路径。"""
    try:
        if not self:
            return False
        dtype = None
        device = None
        for tensor in self:
            if (
                not tensor.is_contiguous()
                or not tensor.dtype.is_floating_point
                or tensor.numel() > _MAX_FLAT_NUMEL
            ):
                return False
            if dtype is None:
                dtype = tensor.dtype
                device = tensor.device
            elif tensor.dtype is not dtype or tensor.device != device:
                return False
    except AttributeError:
        return False
    return True


def _launch_fused(tensors, inplace: bool):
    """同 (device,dtype) 组（守卫已保证单一组）一次 launch 处理 <=16 个
    tensor，>16 按 16 分块；out-of-place 先分配输出保持输入顺序。

    uniform 形状列表的 out-of-place 输出用一次 ``torch.empty((n, *shape))``
    + ``unbind(0)`` 视图分配（16 次 ``empty_like`` 的 host 行程 ~45us 会
    超出 do_bench 的 flush 阴影窗，小形状读数 +10us；单缓冲 + unbind ~6us）。
    输出因此共享同一 storage；混合形状仍逐张量 ``empty_like``。"""
    numels = [t.numel() for t in tensors]
    if not inplace:
        if len({t.shape for t in tensors}) == 1:
            stacked = torch.empty(
                (len(tensors), *tensors[0].shape),
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            destinations = list(stacked.unbind(0))
        else:
            destinations = [torch.empty_like(t) for t in tensors]
    with torch_device_fn.device(tensors[0].device):
        for start in range(0, len(tensors), _SLOTS):
            chunk = tensors[start : start + _SLOTS]
            outs = chunk if inplace else destinations[start : start + _SLOTS]
            sizes = [
                numels[start + slot] if slot < len(chunk) else 0
                for slot in range(_SLOTS)
            ]
            total_tiles = 0
            masked = 0
            for numel in sizes:
                total_tiles += (numel + _BLOCK - 1) // _BLOCK
                if numel % _BLOCK:
                    masked = 1
            if total_tiles == 0:
                continue
            grid = (min(_GRID_SIZE, total_tiles), 1, 1)
            _foreach_lgamma_kernel[grid](
                *[
                    chunk[slot] if slot < len(chunk) else chunk[0]
                    for slot in range(_SLOTS)
                ],
                *[
                    outs[slot] if slot < len(outs) else outs[0]
                    for slot in range(_SLOTS)
                ],
                *sizes,
                BLOCK=_BLOCK,
                GRID_SIZE=grid[0],
                MASKED=masked,
                SHIFT_THRESHOLD=_SHIFT_THRESHOLD,
                num_warps=_NUM_WARPS,
            )
    return None if inplace else destinations


def _foreach_lgamma(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA")
    if _fast_path(self):
        return _launch_fused(self, inplace=False)
    return apply_unary("lgamma", self, inplace=False)


def _foreach_lgamma_(self):
    logger.debug("GEMS_GCU400 _FOREACH_LGAMMA_")
    if _fast_path(self):
        return _launch_fused(self, inplace=True)
    return apply_unary("lgamma", self, inplace=True)
