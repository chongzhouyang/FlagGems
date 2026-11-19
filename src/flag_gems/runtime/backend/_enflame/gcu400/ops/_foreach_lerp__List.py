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
"""GCU400 专属 ``_foreach_lerp_.List``（in-place）：单 launch 融合 kernel。

旧实现逐 tensor 启动一维 kernel（16 张量列表 = 16 次 launch，BLOCK=65536
单 program 串行跨步），16×256² 实测 ~250-280µs，比原生 ATen 慢 ~45%
（官方 bench core：mean 0.55×）。这里按 ``_foreach_pow__List`` 的同型
修法收敛为一次 launch：

1. 均匀浮点同形三元组列表走本文件的融合 kernel：地址/元素数 metadata
   经公共 ``_device_meta`` 只读缓存（稳态调用零同步 H2D），``grid =
   NT * PARTS`` 恒为 tensor 数的整除倍，每个 tensor 恰有 ``PARTS``
   个 program 以 ``PARTS`` 为步长分摊 tile——``(part, tensor)`` 线性
   化在 grid 不整除 tensor 数时会漏写部分 tile（24 % 16 != 0 时
   tensor 8-15 只分到 part 0，stride 2 跳过奇数 tile），本路径按构造
   排除该形态；计算保持现行 fp32 提升语义（同 ATen opmath 与家族
   ``fused_lerp_scalar_list_kernel``）；fp32 且全部 numel 整除 BLOCK 时
   走 EXACT 满块免 mask 臂（同几何 -3~-15%），16-bit 与 ragged 走
   masked 臂。
2. 非浮点、非连续、混合 dtype、广播 weight、跨设备等形态回退公共
   ``_apply_lerp``，保持 dtype policy 与异常语义不变。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_binary import _apply_lerp
from flag_gems.utils import libentry
from flag_gems.utils.foreach import _device_meta

from ._foreach_flat import _TL_DTYPES

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 由优化迭代驱动脚本按轮次生成；编译期/启动期配置，不改变算子语义。
# 快路径几何（本机新栈 Triton GCU 4.1.6，P{1..6}xB{4096..32768}xwxs
# 交错扫描 + P{2,3,4}xB{4096,8192,16384} 精扫，逐组合 16 张量全量
# 覆盖校验）：P3B8192w4s1 聚合最优（fp16 全档胜出、fp32 与 B4096 档
# 差 3-16% 但聚合归一 1.04 仍最优）；P3B32768 沿 pow 2R1W 制式反劣
# 5-15%（3R1W 行为与家族 ScalarList fused 48-grid/B8192 制式一致），
# w8/s2 在噪声带内取最简。
# 2026-11-19 P3 增量轮：EXACT 免 mask 臂（maximum_List 判例，(dtype,BLOCK)
# 耦合）——fp32 在 B8192 同几何下 -14.8/-6.9/-3.1%（256²/512²/1024²），
# 16-bit EXACT 中性偏负（+0.3~2.3%）且 4 流 B16384 反劣化 +8.7~19.3%
# （addcmul 判例：maximum_List 16-bit B16384 收益系 2R1W 专属）；fp32
# EXACT 臂 P6 +1.1~12.6%、nw2/B16384 持平——故 EXACT 仅对 fp32 且全部
# numel 整除 BLOCK 时启用，16-bit 与 ragged 维持 masked 同几何。
FAST_PARTS_CAP = 3
FAST_BLOCK = 8192
FAST_WARPS = 4
FAST_STAGES = 1
FP32_PROMOTE = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

# GCU400 hard limit on grid.x; the fast-path grid = nt * parts must fit entirely within the limit.
_MAX_GRID_X = 65535

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)


@libentry()
@triton.jit
def fused_lerp_list_ip_kernel(
    meta_ptr,
    NT: tl.constexpr,
    A_DT: tl.constexpr,
    FP32_PROMOTE: tl.constexpr,
    EXACT: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
):
    """一次 launch 对整个三元组列表做 in-place ``self + w*(end-self)``。

    ``meta_ptr`` 依次存放输入、end、weight、输出（in-place 下即输入）
    地址与各元素数（各 NT 项，int64）。grid 按 ``(tensor, part)`` 线性
    展开且恒为 ``NT * PARTS``：每个 tensor 的 ``PARTS`` 个 program 以
    ``PARTS`` 为步长分摊 tile，覆盖洞按构造不存在。

    ``EXACT`` 为真时每个 tensor 的元素数都整除 ``BLOCK``（host 侧保证），
    全部 tile 都是满块，load/store 免 mask 谓词（fp32 实测 -3~-15%，
    16-bit 无收益故仅 fp32 启用）；否则走 masked 保语义。
    """
    pid = tl.program_id(0)
    tensor_id = pid % NT
    part_id = pid // NT

    a_ptr = tl.load(meta_ptr + tensor_id).to(tl.pointer_type(A_DT))
    b_ptr = tl.load(meta_ptr + NT + tensor_id).to(tl.pointer_type(A_DT))
    w_ptr = tl.load(meta_ptr + 2 * NT + tensor_id).to(tl.pointer_type(A_DT))
    out_ptr = tl.load(meta_ptr + 3 * NT + tensor_id).to(tl.pointer_type(A_DT))
    n_elements = tl.load(meta_ptr + 4 * NT + tensor_id)

    arange = tl.arange(0, BLOCK)
    # tile 计数保持 i32，与公共执行器的地址算术宽度一致。
    if EXACT:
        num_tiles = (n_elements // BLOCK).to(tl.int32)
    else:
        num_tiles = tl.cdiv(n_elements, BLOCK).to(tl.int32)
    for block_id in tl.range(part_id, num_tiles, PARTS):
        idx = block_id * BLOCK + arange
        if EXACT:
            a = tl.load(a_ptr + idx)
            b = tl.load(b_ptr + idx)
            w = tl.load(w_ptr + idx)
            if FP32_PROMOTE:
                # 低精度先提升 fp32 再算（ATen opmath 语义），store 前降回。
                a = a.to(tl.float32)
                b = b.to(tl.float32)
                w = w.to(tl.float32)
            tl.store(out_ptr + idx, (a + w * (b - a)).to(A_DT))
        else:
            mask = idx < n_elements
            a = tl.load(a_ptr + idx, mask=mask)
            b = tl.load(b_ptr + idx, mask=mask)
            w = tl.load(w_ptr + idx, mask=mask)
            if FP32_PROMOTE:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
                w = w.to(tl.float32)
            tl.store(out_ptr + idx, (a + w * (b - a)).to(A_DT), mask=mask)


def _run_lerp_list_cached(self, tensors1, weight):
    """均匀浮点同形三元组列表的单 launch in-place 路径。

    不满足前提（广播 weight、跨设备、非连续、混合 dtype、超 grid 上限）
    时返回 ``FALLBACK``，由调用方回退公共执行器；metadata 只含地址与
    元素数，kernel 每次仍完整读取并计算张量数据，缓存仅省去每次调用
    的同步 H2D。空张量的 tile 数为 0，核内循环自然空转，语义不变。
    """
    if FALLBACK_TO_PUBLIC:
        return False
    if not isinstance(self, (list, tuple)) or not self:
        return False
    if not isinstance(tensors1, (list, tuple)) or not isinstance(weight, (list, tuple)):
        return False
    if len(tensors1) != len(self) or len(weight) != len(self):
        return False
    num_tensors = len(self)
    if num_tensors * FAST_PARTS_CAP > _MAX_GRID_X:
        return False
    first = self[0]
    if not isinstance(first, torch.Tensor) or first.dtype not in _FLOAT_DTYPES:
        return False
    dtype = first.dtype
    device = first.device
    for a, b, w in zip(self, tensors1, weight):
        if not (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and isinstance(w, torch.Tensor)
        ):
            return False
        if a.dtype is not dtype or b.dtype is not dtype or w.dtype is not dtype:
            return False
        if a.device != device or b.device != device or w.device != device:
            return False
        if not (a.is_contiguous() and b.is_contiguous() and w.is_contiguous()):
            return False
        # Broadcasting (including a 0-d weight) doesn't share flat indices with the input, so fall back to the common
        # executor.
        if b.shape != a.shape or w.shape != a.shape:
            return False

    in_ptrs = [t.data_ptr() for t in self]
    b_ptrs = [t.data_ptr() for t in tensors1]
    w_ptrs = [t.data_ptr() for t in weight]
    numels = [t.numel() for t in self]
    meta = _device_meta(in_ptrs + b_ptrs + w_ptrs + in_ptrs + numels, device)
    parts = max(1, min(FAST_PARTS_CAP, triton.cdiv(max(numels), FAST_BLOCK)))
    # EXACT 臂仅 fp32 且全部 numel 整除 BLOCK：满块免 mask 谓词实测
    # -3~-15%；16-bit 无收益（(dtype,BLOCK) 耦合判例）、ragged 保 masked。
    use_exact = dtype == torch.float32 and all(n % FAST_BLOCK == 0 for n in numels)
    fused_lerp_list_ip_kernel[(num_tensors * parts,)](
        meta,
        num_tensors,
        _TL_DTYPES[dtype],
        FP32_PROMOTE,
        use_exact,
        FAST_BLOCK,
        parts,
        num_warps=FAST_WARPS,
        num_stages=FAST_STAGES,
    )
    return True


def _foreach_lerp__List(self, tensors1, weight):
    logger.debug("GEMS_GCU400 FOREACH_LERP__LIST")
    if not _run_lerp_list_cached(self, tensors1, weight):
        return _apply_lerp(self, tensors1, weight, True)
    # in-place schema contract: the dispatcher side must return None.
    return None
