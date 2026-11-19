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

"""GCU400 专属的 W8A16 FP8 TopK.

第一版位解码实现（单 kernel K 轮 max/argmax、大 N 两阶段 chunk+merge）在
benchmark 口径下平均加速比仅 ~0.45x：K 轮选路的每轮成本是归约树延迟
（~0.2-1.2us/轮，随宽度与驻留形态劣化），K=64..256 时 O(K*N) 选路链把
(8,32768,256) 推到 738us（torch native 24.7us）。本轮按形状分派三路：

1. ``_multi_row_kernel``：N 较小时一行 K 轮改为 **U 行交织**（同相批排：
   全部行 max -> 全部行 min-pos -> 全部行 invalidate），独立链互相掩盖
   归约树延迟，U 按「ceil(M/U) 波 = 24 SIP」+「U*BLOCK_N <= 2816 lane
   预算」选档；比较域换成 fp32 的单调 int32 key（两棵纯归约替代
   max+argmax，且 -0.0/+0.0 顺序与 torch 位模式序一致）。
2. ``_dequant_bf16_kernel`` + 兄弟模块 ``topk_mradix``：大 N 走「fp8 解码
   成 bf16 行 -> 硬件 MRADIX radix-select」组合（FlagGems 自己的 kernel，
   单次 HBM 读 + VDMEM 数字扫描，实测 22-38us 覆盖 N=4k..32k 档）。
   largest=False 通过解码时翻转 fp8 符号位（= 值取负）+ 事后小 negate
   kernel 还原；参考域同为 bf16 dequant，值域精确一致。
3. 既有单 kernel / 两阶段 chunk+merge 保留为 MRADIX 包络外回退
   （N>65536、K>512、非 bf16 out_dtype 等）。
"""

import logging
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

try:
    # Compositional reuse of the sibling module's hardware MRADIX topk (TLE-Struct + TLE-Raw). If the import fails
    # (no .tops file / no TLE-Raw binding), silently fall back to the old path.
    from . import topk_mradix as _mradix

    _MRADIX_OK = bool(_mradix.HAS_TLE_RAW)
except Exception:  # noqa: BLE001
    _mradix = None
    _MRADIX_OK = False

# Single-block column width: fp32 candidate live set 4096 * 4B = 16KB, consistent with the rms_norm tile budget.
_CHUNK = 4096
# Candidate buffer cap (NUM_CHUNK * K per row); beyond it, increase the block width to reduce the number of blocks.
_CAND_CAP = 8192
# Minimum reduction amount (number of blocks * K) to enable two-stage; below it, use a single kernel (saves one launch).
_CHUNK_MIN_REDUCTIONS = 1024
# Beyond this block width, abandon two-stage and fall back to a single kernel (avoid ultra-wide vectors entering the DSM
# wrong-value region).
_MAX_CHUNK = 65536
_CHUNK_WARPS = 4
_MERGE_WARPS = 1
_SINGLE_WARPS = 1

# Settings for row-interleaving over K rounds (2026-11-18 U-sweep measurements: with U=1 and no interleaving, the
# dependency chain per round
# takes ~500-700ns; with U>=2 interleaving, ~110-170ns (7-20x); optimal U = the smallest power of 2 that fills one full
# wave (24 SIPs),
# with U*BLOCK_N capped at 8192 lane ((128,1024,16) with U=8 measured best).
_KR_WAVES = 24
_KR_LANE_BUDGET = 8192
# Cost model per round per row (us): <=1024 width resident ~0.28, wider scales x2 by byte ratio (spill penalty).
_KR_ROUND_US = 0.28
# MRADIX per-row cost model (us): fixed ~9 + N * 0.7ns (VDMEM digit scan).
_MR_ROW_FIXED_US = 9.0
_MR_ROW_PER_ELEM_NS = 0.0007
# mradix envelope (consistent with topk_mradix.TOPK_K_MAX / SORT_K_MAX / L1_MAX_COLS).
_MR_K_MAX = 512
_MR_N_MAX = 65536


@triton.jit
def _decode_fp8(bits, E4M3: tl.constexpr):
    """FP8 位模式 -> float32（含 subnormal；不依赖设备侧 FP8 类型）。"""
    sign = (bits >> 7) & 1
    if E4M3:
        exp = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        normal_bits = (sign << 31) | ((exp + 120) << 23) | (mantissa << 20)
        subnormal = mantissa.to(tl.float32) * 0.001953125
    else:
        exp = (bits >> 2) & 0x1F
        mantissa = bits & 0x3
        normal_bits = (sign << 31) | ((exp + 112) << 23) | (mantissa << 21)
        subnormal = mantissa.to(tl.float32) * 1.52587890625e-5

    # The IEEE bits of a normal already include the sign bit; a subnormal has no bit pattern, so the sign must be
    # carried separately
    normal_value = normal_bits.to(tl.float32, bitcast=True)
    signed_subnormal = tl.where(sign == 1, -subnormal, subnormal)
    return tl.where(exp > 0, normal_value, signed_subnormal)


@triton.jit
def _group_index(cols, GROUP_SIZE: tl.constexpr, LOG_GROUP: tl.constexpr):
    """组号：2 的幂时移位，否则退回整除（GCU 无整数除法硬件，逐 lane 展开很贵）。"""
    if LOG_GROUP > 0:
        return cols >> LOG_GROUP
    else:
        return cols // GROUP_SIZE


@triton.jit
def _mono_key(value):
    """fp32 -> 单调 int32 key（自身可逆）。

    bits>=0 时 key=bits，否则 key=bits^0x7FFFFFFF：负数区间被反向映射到
    [-2^31, -1]，任意两值的大小序与 key 序一致，且 -0.0 < +0.0（与
    torch.topk 的位模式序一致）。翻转符号（~key）即得降序域。
    """
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)


@triton.jit
def _unmono_key(key):
    bits = tl.where(key >= 0, key, key ^ 0x7FFFFFFF)
    return bits.to(tl.float32, bitcast=True)


_KEY_SENTINEL = tl.constexpr(-2147483648)


@libentry()
@triton.jit
def topk_w8a16_fp8_multi_row_kernel(
    x_ptr,
    scale_ptr,
    y_vals_ptr,
    y_idx_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    U: tl.constexpr,
):
    """U 行交织的 K 轮选路：同相批排掩盖归约树延迟。

    每轮对 [U, BLOCK_N] 的 int32 key 域做：全部行 max -> 全部行
    min-position（同值取最小列）-> 向量 store -> 全部行 invalidate。
    """
    pid = tl.program_id(0)
    rows = pid * U + tl.arange(0, U)
    live = rows < M
    cols = tl.arange(0, BLOCK_N)
    valid = (cols < N)[None, :] & live[:, None]
    bits = tl.load(x_ptr + rows[:, None] * N + cols[None, :], mask=valid, other=0).to(
        tl.int32
    )
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(cols, GROUP_SIZE, LOG_GROUP)[None, :]
    scale = tl.load(scale_ptr + rows[:, None] * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    key = _mono_key(decoded * scale)
    if not LARGEST:
        key = ~key
    cand = tl.where(valid, key, _KEY_SENTINEL)

    for i in range(K):
        best = tl.max(cand, axis=1)
        pos = tl.min(tl.where(cand == best[:, None], cols[None, :], BLOCK_N), axis=1)
        if not LARGEST:
            # After flipping the key domain, max selects the smallest-value element; flipping back to the original
            # domain restores the true value
            best = ~best
        vals = _unmono_key(best)
        tl.store(
            y_vals_ptr + rows * K + i, vals.to(y_vals_ptr.dtype.element_ty), mask=live
        )
        tl.store(y_idx_ptr + rows * K + i, pos.to(tl.int64), mask=live)
        cand = tl.where(cols[None, :] == pos[:, None], _KEY_SENTINEL, cand)


@libentry()
@triton.jit
def topk_w8a16_fp8_kernel(
    x_ptr,
    scale_ptr,
    y_vals_ptr,
    y_idx_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N
    bits = tl.load(x_ptr + row * N + offs, mask=valid, other=0).to(tl.int32)
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(offs, GROUP_SIZE, LOG_GROUP)
    scale = tl.load(scale_ptr + row * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    value = decoded * scale

    # Mask invalid padding with extrema so extrema / indices are selected only among real elements
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    candidates = tl.where(valid, value, sentinel)

    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
            best_index = tl.argmax(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
            best_index = tl.argmin(candidates, 0)
        tl.store(y_vals_ptr + row * K + i, best_value)
        tl.store(y_idx_ptr + row * K + i, best_index.to(tl.int64))
        candidates = tl.where(offs == best_index, sentinel, candidates)


@triton.jit
def topk_w8a16_fp8_chunk_kernel(
    x_ptr,
    scale_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    N,
    K: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    LARGEST: tl.constexpr,
    E4M3: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOG_GROUP: tl.constexpr,
    NUM_CHUNK: tl.constexpr,
):
    """阶段 1：一个 program 负责 (行, 块)，产出该块的 top-K 候选。"""
    pid = tl.program_id(0)
    row = pid // NUM_CHUNK
    chunk = pid - row * NUM_CHUNK
    offs = tl.arange(0, BLOCK_N)
    cols = chunk * BLOCK_N + offs
    valid = cols < N

    bits = tl.load(x_ptr + row * N + cols, mask=valid, other=0).to(tl.int32)
    decoded = _decode_fp8(bits, E4M3)
    group = _group_index(cols, GROUP_SIZE, LOG_GROUP)
    scale = tl.load(scale_ptr + row * GROUPS + group, mask=valid, other=0).to(
        tl.float32
    )
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    candidates = tl.where(valid, decoded * scale, sentinel)

    # Candidates are packed tightly in program order: pid * K + i == row*NUM_CHUNK*K + chunk*K + i
    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
        # For equal values, take the smallest lane, ensuring that together with the later "hole punching" the same
        # element is never selected twice
        pos = tl.min(tl.where(candidates == best_value, offs, BLOCK_N), 0)
        tl.store(cand_val_ptr + pid * K + i, best_value)
        tl.store(cand_idx_ptr + pid * K + i, (chunk * BLOCK_N + pos).to(tl.int32))
        candidates = tl.where(offs == pos, sentinel, candidates)


@triton.jit
def topk_w8a16_fp8_merge_kernel(
    cand_val_ptr,
    cand_idx_ptr,
    y_vals_ptr,
    y_idx_ptr,
    K: tl.constexpr,
    LARGEST: tl.constexpr,
    BLOCK_C: tl.constexpr,
    NUM_CAND: tl.constexpr,
):
    """阶段 2：对每行的 NUM_CAND 个候选再取 K 次极值，输出有序 top-K。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    valid = offs < NUM_CAND
    if LARGEST:
        sentinel = float("-inf")
    else:
        sentinel = float("inf")
    values = tl.load(cand_val_ptr + row * NUM_CAND + offs, mask=valid, other=sentinel)
    indices = tl.load(cand_idx_ptr + row * NUM_CAND + offs, mask=valid, other=-1)
    candidates = tl.where(valid, values, sentinel)

    for i in range(K):
        if LARGEST:
            best_value = tl.max(candidates, 0)
        else:
            best_value = tl.min(candidates, 0)
        pos = tl.min(tl.where(candidates == best_value, offs, BLOCK_C), 0)
        # Retrieve the true column index of this candidate (scalar reduction, only one value is moved)
        best_index = tl.sum(tl.where(offs == pos, indices, 0), 0)
        tl.store(y_vals_ptr + row * K + i, best_value)
        tl.store(y_idx_ptr + row * K + i, best_index.to(tl.int64))
        candidates = tl.where(offs == pos, sentinel, candidates)


@libentry()
@triton.jit
def topk_w8a16_fp8_dequant_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    N,
    GROUPS,
    GROUP_SIZE: tl.constexpr,
    E4M3: tl.constexpr,
    NEGATE: tl.constexpr,
    GT: tl.constexpr,
):
    """fp8+组 scale -> bf16 行（MRADIX 组合路径的阶段 1）。

    以 [GT, GROUP_SIZE] 的 2D 组 tile 访存：x/out 走仿射连续块，scale 只做
    [GT] 一维 load 后广播——避免逐元素 gather 型 scale load（GCU 上逐元素
    发射 ~10ns/元素，2026-11-18 实测把本 kernel 拖到 ~21us）。
    NEGATE 时翻转 fp8 符号位（等价于值取负），供 largest=False 借道
    MRADIX 的 largest-only radix-select。要求 group_size 为 2 的幂（host 侧
    门控）。
    """
    row = tl.program_id(0)
    tile = tl.program_id(1)
    g = tile * GT + tl.arange(0, GT)
    gvalid = g < GROUPS
    cols = tl.arange(0, GROUP_SIZE)
    inc = g[:, None] * GROUP_SIZE + cols[None, :]
    valid = inc < N
    bits = tl.load(x_ptr + row * N + inc, mask=valid, other=0).to(tl.int32)
    if NEGATE:
        bits = bits ^ 0x80
    decoded = _decode_fp8(bits, E4M3)
    scale = tl.load(scale_ptr + row * GROUPS + g, mask=gvalid, other=0).to(tl.float32)
    val = decoded * scale[:, None]
    tl.store(out_ptr + row * N + inc, val.to(out_ptr.dtype.element_ty), mask=valid)


@libentry()
@triton.jit
def topk_w8a16_fp8_negate_kernel(
    y_ptr,
    total,
    BLOCK: tl.constexpr,
):
    """把 (M, K) 值张量按元素取负（largest=False 借道后的还原）。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < total
    vals = tl.load(y_ptr + offs, mask=valid)
    tl.store(y_ptr + offs, -vals, mask=valid)


def _log2_if_pow2(value):
    """2 的幂返回 log2，否则返回 0（kernel 侧据此选择移位或整除）。"""
    if value <= 0 or (value & (value - 1)) != 0:
        return 0
    return value.bit_length() - 1


def _kround_rows(m, block_n, k):
    """行交织 K 轮的 U 选档。

    U = 满 1 波（ceil(M/U) <= 24 SIP）的最小 2 的幂，下限 2（U=1 无交织时
    依赖链每轮退化 4-7 倍，2026-11-18 实测）；U*BLOCK_N 超 8192 lane 则
    减半防 spill。返回 (U, 预估 makespan us)。
    """
    u = 2
    while u * _KR_WAVES < m:
        u *= 2
    while u > 2 and u * block_n > _KR_LANE_BUDGET:
        u //= 2
    programs = -(-m // u)
    waves = max(1, -(-programs // _KR_WAVES))
    per_round = _KR_ROUND_US if block_n <= 1024 else _KR_ROUND_US * block_n / 1024 * 2
    return u, waves * u * k * per_round


def _mradix_rows_us(m, n):
    waves = max(1, -(-m // _KR_WAVES))
    return waves * (_MR_ROW_FIXED_US + n * _MR_ROW_PER_ELEM_NS)


def _chunk_plan(n, k):
    """返回 (块宽, 块数)；不划算或块宽超限时返回 (None, 1) 走单 kernel。

    实测（2026-11-17，gcu400）两阶段的总耗时近似正比于「块数 × K」次归约，
    并额外付一次 launch；因此只在归约量足够大（``块数 × K >= 1024``）时才启用：

    * (8,32768,256)：块数 8 × 256 = 2048 → 两阶段 1.22ms（单 kernel 1.62ms）；
    * (16,16384,128)：4 × 128 = 512 → 单 kernel 0.58ms 反而更快（两阶段 0.59ms）；
    * (32,8192,64)：2 × 64 = 128 → 单 kernel 0.36ms。

    块宽只在候选缓冲超 ``_CAND_CAP`` 时才加倍。
    """
    chunk = _CHUNK
    while True:
        num_chunk = triton.cdiv(n, chunk)
        if num_chunk * k <= _CAND_CAP or chunk >= _MAX_CHUNK:
            break
        chunk *= 2
    if num_chunk <= 1 or chunk > _MAX_CHUNK or num_chunk * k < _CHUNK_MIN_REDUCTIONS:
        return None, 1
    return chunk, num_chunk


def _topk_kround(
    x_bits,
    scale_2d,
    y_vals,
    y_idx,
    m,
    n,
    k,
    num_groups,
    group_size,
    largest,
    is_e4m3,
    log_group,
):
    block_n = triton.next_power_of_2(n)
    u, _ = _kround_rows(m, block_n, k)
    with torch_device_fn.device(x_bits.device):
        topk_w8a16_fp8_multi_row_kernel[(-(-m // u),)](
            x_bits,
            scale_2d,
            y_vals.reshape(m, k),
            y_idx.reshape(m, k),
            m,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            block_n,
            log_group,
            u,
            num_warps=1,
        )
    return y_vals, y_idx


def _topk_mradix_compose(
    x_bits,
    scale_2d,
    y_vals,
    y_idx,
    m,
    n,
    k,
    num_groups,
    group_size,
    largest,
    is_e4m3,
    log_group,
):
    """fp8 解码成 bf16 行 -> 硬件 MRADIX topk（largest=False 借道取负）。

    直接调兄弟模块的 launcher 而非其 python 封装：跳过内部的 i32 索引张量
    与 `.to(int64)` aten cast（一记 kernel + 分发开销），i64 输出由 launcher
    的 store 隐式宽化直接落盘。
    """
    with torch_device_fn.device(x_bits.device):
        dequant = torch.empty((m, n), device=x_bits.device, dtype=torch.bfloat16)
        # GT adaptation: larger tiles amortize the per-program fixed cost (2026-11-18 measurements:
        # GT 8/32/128 = 30/12.5/10.5us@N=32768), capped at 128 groups to avoid register pressure.
        gt = min(128, triton.next_power_of_2(num_groups))
        topk_w8a16_fp8_dequant_kernel[(m, triton.cdiv(num_groups, gt))](
            x_bits,
            scale_2d,
            dequant,
            n,
            num_groups,
            group_size,
            is_e4m3,
            not largest,
            gt,
            num_warps=4,
        )
        grid = min(_mradix._max_grid(x_bits.device.index), m)
        _mradix.topk_mradix_l1_launch[(grid,)](
            dequant,
            y_vals.reshape(m, k),
            y_idx.reshape(m, k),
            m,
            n,
            K=k,
            K_PAD=triton.next_power_of_2(k),
            BLOCK_N=max(256, triton.next_power_of_2(n)),
            ROW_STEP=1,
            SORTED=1,
            IN_KIND=2,
            OUT_KIND=2,
            num_warps=4,
            num_stages=1,
        )
        if not largest:
            topk_w8a16_fp8_negate_kernel[(triton.cdiv(m * k, 1024),)](
                y_vals, m * k, 1024, num_warps=4
            )
    return y_vals, y_idx


def topk_w8a16_fp8(
    x_fp8,
    x_scale,
    k,
    dim=-1,
    largest=True,
    sorted=True,
    group_size=128,
    out_dtype=torch.bfloat16,
):
    logger.debug("GEMS_ENFLAME TOPK_W8A16_FP8")
    if dim < 0:
        dim += x_fp8.ndim
    assert dim == x_fp8.ndim - 1, "Currently only support topk in last dimension"
    assert sorted, "Currently only support sorted == True"

    leading_shape = x_fp8.shape[:-1]
    n = x_fp8.shape[-1]
    if k == 0:
        out_shape = list(leading_shape) + [0]
        return (
            torch.empty(out_shape, device=x_fp8.device, dtype=out_dtype),
            torch.empty(out_shape, device=x_fp8.device, dtype=torch.int64),
        )

    batch_size = math.prod(leading_shape) if leading_shape else 1
    num_groups = triton.cdiv(n, group_size)
    expected_scale_shape = tuple(leading_shape) + (num_groups,)
    assert (
        x_scale.shape == expected_scale_shape
    ), f"x_scale shape should be {expected_scale_shape}, got {x_scale.shape}"

    # Read FP8 bit patterns as uint8: does not rely on device-side FP8 pointers, nor does it modify the quantized data
    x_bits = x_fp8.contiguous().view(torch.uint8).reshape(batch_size, n)
    scale_2d = x_scale.reshape(batch_size, num_groups)
    is_e4m3 = x_fp8.dtype == torch.float8_e4m3fn
    log_group = _log2_if_pow2(group_size)

    def alloc_out():
        return (
            torch.empty(
                tuple(leading_shape) + (k,), device=x_fp8.device, dtype=out_dtype
            ),
            torch.empty(
                tuple(leading_shape) + (k,), device=x_fp8.device, dtype=torch.int64
            ),
        )

    # Path 1: row-interleaved K rounds (small N / small K: the O(K) selection chain is fastest at resident width).
    if n <= _KR_LANE_BUDGET:
        block_n = triton.next_power_of_2(n)
        _, est_kround = _kround_rows(batch_size, block_n, k)
        use_kround = est_kround <= _mradix_rows_us(batch_size, n) + 4.0
    else:
        use_kround = False
    if use_kround:
        y_vals, y_idx = alloc_out()
        return _topk_kround(
            x_bits,
            scale_2d,
            y_vals,
            y_idx,
            batch_size,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            log_group,
        )

    # Path 2: dequant + hardware MRADIX (workhorse for large N; K/N within the envelope, bf16 output,
    # group_size a power of two — required by dequant's 2D group-tile memory access).
    if (
        _MRADIX_OK
        and 0 < k <= _MR_K_MAX
        and n <= _MR_N_MAX
        and out_dtype == torch.bfloat16
        and log_group > 0
    ):
        y_vals, y_idx = alloc_out()
        return _topk_mradix_compose(
            x_bits,
            scale_2d,
            y_vals,
            y_idx,
            batch_size,
            n,
            k,
            num_groups,
            group_size,
            largest,
            is_e4m3,
            log_group,
        )

    # Path 3 (fallback): single kernel / two-stage chunk+merge.
    y_vals, y_idx = alloc_out()
    with torch_device_fn.device(x_fp8.device):
        chunk, num_chunk = _chunk_plan(n, k)
        if chunk is None:
            topk_w8a16_fp8_kernel[(batch_size,)](
                x_bits,
                scale_2d,
                y_vals.reshape(batch_size, k),
                y_idx.reshape(batch_size, k),
                n,
                k,
                num_groups,
                group_size,
                largest,
                is_e4m3,
                triton.next_power_of_2(n),
                log_group,
                num_warps=_SINGLE_WARPS,
            )
        else:
            num_cand = num_chunk * k
            cand_val = torch.empty(
                (batch_size * num_cand,), device=x_fp8.device, dtype=torch.float32
            )
            cand_idx = torch.empty(
                (batch_size * num_cand,), device=x_fp8.device, dtype=torch.int32
            )
            topk_w8a16_fp8_chunk_kernel[(batch_size * num_chunk,)](
                x_bits,
                scale_2d,
                cand_val,
                cand_idx,
                n,
                k,
                num_groups,
                group_size,
                largest,
                is_e4m3,
                chunk,
                log_group,
                num_chunk,
                num_warps=_CHUNK_WARPS,
            )
            topk_w8a16_fp8_merge_kernel[(batch_size,)](
                cand_val,
                cand_idx,
                y_vals.reshape(batch_size, k),
                y_idx.reshape(batch_size, k),
                k,
                largest,
                triton.next_power_of_2(num_cand),
                num_cand,
                num_warps=_MERGE_WARPS,
            )
    return y_vals, y_idx
