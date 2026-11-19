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
"""Enflame gcu400 fused AdamW：持久化 meta 表单 launch 策略。

新栈（flagtree 0.7.0+enflame.gitd8ca7a9e / Triton GCU 4.1.6，2026-11-18）
实测归因与上一轮（2026-11-16，commit 24822709f）结论合并如下：

- **大张量主瓶颈是波序列化，不是块形状**：同一份公共 kernel，BLOCK 4096
  全量 grid（4096 个 CTA）f16 单张量 (2**24,) 273.7µs；改持久化 grid-stride
  （48 个 CTA，每 program 步进 chunk）后 149.0µs（-46%），逐块扫 48/64/96/
  128/160/192 中 48（单 wave）最优。i64 向量偏移、masked 谓词、exact 去
  mask 三者全量 grid 下差异为 0（274.0/274.6/274.5µs）＝纯持久化收益。
- **num_warps（4/8/16）与 enable_fp_fusion 在该 kernel 上零差异**（273.7-
  273.9µs 级带宽主导），后端忽略，不再作为调优维度。
- **多张量主瓶颈是 host**：many_small_64（64×257）走公共实现 host 443µs
  （逐张量 Python 校验 + 524 参 binder ~120µs/launch），many_small_256 高达
  1710-1850µs；官方 bench（do_bench kernel 模式）约 72-90µs 的 256MB flush
  阴影窗只能隐藏低于该预算的 host 时间。

因此本模块把整条路径换成「一次持久化 launch」：

1. 校验用 C 级批量原语（``attrgetter`` 单遍 + ``map`` 集/布尔聚合）完成公
   共实现的 dtype/device/shape/连续性/state_steps 契约；任何不满足原样回退
   公共实现，由它给出原有报错，且校验完成前不 launch 任何 kernel。
2. 指针/元素数打成一张 int64 meta 表（group-major：6×NT 指针行 + NT 数量
   行），经 ``flag_gems.utils.foreach._device_meta`` 的 **值键** memo 复用
   （指针值变即重建，稳态零 H2D）；每次调用重读全部 ``data_ptr``，不缓存
   张量身份或地址。
3. launch 走 ``_foreach_reduction._launch`` 的 cached-launcher replay（同
   ``_foreach_zero`` 等已验收制式）；lr/beta 等 9 个标量是
   ``do_not_specialize`` 运行时实参，replay 每次传当前值，lr 调度/设备侧
   lr 均正确。
4. kernel 的标量前导（step 读出与偏差校正多项式）复用公共
   ``_adamw_correction``；逐元素体是公共 fp32 路径的持久化寻址改写（块内
   i32 向量 + 标量 i64 块基址；A/B 实测逐块原样复用公共体在持久化形态下
   318µs vs 本形态 149µs@f16 (2**24,)）。主循环只跑整块、单一份循环体，
   尾块（若有）在循环后按 mask 处理一次——逐块内联双分支（整块/masked 两
   份体）在 amsgrad f32 上把寄存器压力推过一档（345 vs 232µs）。
   AMSGrad 的 max 二阶矩 store 移到 param/moments 之后（A/B -30%：
   f32 ams1 333→232µs、f16 ams1 210→167µs；数值不变，仅别名病态输入
   下最终内存态与公共顺序不同，公共 API 本不支持别名）。
5. 多张量不均负载均衡：(NT, 48) 每张量固定 48 program，混合尺寸下
   2:1 失衡（hetero f16 104.6µs）；按 chunk 占比分配 program 数、每
   program 行 (tensor, 起始块, 块内步进) 写入 meta（target 192）后
   81.5µs（f16 ams0，-22%），f16/f32 全档最优（48/96/192 扫描）。
   meta 指针还原（``int64 -> tl.pointer_type``）同 ``_foreach_zero``
   判例，依赖 ``TORCH_GCU_ENABLE_INT64_AND_UINT64``（默认全局开启；本模块
   直接读该环境变量，未开启即回退公共实现——静态 ``support_int64`` 能力
   位不追踪该开关，env-off 下 int64 H2D 会直接 abort 进程）。

``_fused_adamw_`` / ``_fused_adamw__tensor_lr`` 仍导出公共模块的同一函数对
象；公共函数按模块全局名解析 ``_fused_adamw_run``，本模块替换该全局名，因
此 ATen dispatch 追踪与 tests 的 monkeypatch 均不受影响。
"""

import importlib
import math
import numbers
import operator
import os

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils.foreach import _device_meta, tl_dtype

from ._foreach_reduction import _launch

# Public implementation: reused for the kernel math, the numeric path and as
# the fallback entry for every input the fast path does not accept.
_IMPL = importlib.import_module("flag_gems.ops._fused_adamw_")
_ORIGINAL_RUN = _IMPL._fused_adamw_run
_adamw_correction = _IMPL._adamw_correction

# --- TUNING PARAMS START ---
# Launch shape only; the update math is the shared public implementation.
_GCU400_BLOCK_CAP = 4096  # elements per program chunk (public path uses 1024)
_GCU400_PERSISTENT = 48  # programs per tensor axis; one GCU400 wave
_GCU400_PLAN_TARGET = 192  # total programs for the balanced uneven-size grid
_GCU400_MAX_LIST = 65535  # grid.x hardware cap; larger lists fall back
_GCU400_FAST_PATH = True  # False disables the fast path (A/B control)
# --- TUNING PARAMS END ---

_FAST_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DATA_PTR = torch.Tensor.data_ptr
_NUMEL = torch.Tensor.numel
_IS_CONTIGUOUS = torch.Tensor.is_contiguous
_MAIN_DESC = operator.attrgetter("dtype", "device")
_SHAPE = operator.attrgetter("shape")
_STEP_DESC = operator.attrgetter("dtype", "device", "layout")
_STREAM_CAPTURING = getattr(torch_device_fn, "is_current_stream_capturing", None)


@triton.jit(
    do_not_specialize=[
        "lr",
        "beta1",
        "beta2",
        "one_minus_beta1",
        "one_minus_beta2",
        "log_beta1",
        "log_beta2",
        "weight_decay",
        "eps",
    ]
)
def _adamw_meta_kernel(
    Meta,
    Lr,
    lr,
    beta1,
    beta2,
    one_minus_beta1,
    one_minus_beta2,
    log_beta1,
    log_beta2,
    weight_decay,
    eps,
    TENSOR_LR: tl.constexpr,
    AMSGRAD: tl.constexpr,
    MAXIMIZE: tl.constexpr,
    HAS_PLANS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NT: tl.constexpr,
    PARAM_DT: tl.constexpr,
):
    """Update every tensor of one list in a single persistent launch.

    ``Meta`` is a read-only int64 table, group-major: six pointer rows of NT
    entries (Param, Grad, ExpAvg, ExpAvgSq, MaxExpAvgSq, Step) followed by NT
    element counts and, when ``HAS_PLANS``, P rows of (tensor, first chunk,
    chunk stride) for the balanced multi-tensor grid.

    Without plans the grid is ``(NT, CY)``: axis 0 selects the tensor, axis 1
    is the persistent chunk range and every program strides chunks by CY.
    With plans the grid is ``(P,)`` and each program reads its assignment, so
    uneven tensor sizes still fill the device evenly (2026-11-18 A/B: the
    (NT, 48) form leaves 2:1 program imbalance on the benchmark's mixed
    list at 104.6us against 81.5us for the balanced plan rows).

    The update body is the public fp32 path with the addressing adapted for
    the persistent loop (reusing the public per-block body verbatim costs
    318us on (2**24,) f16 against 149us here; i64 vector offsets plus masked
    predication only tax the persistent form).  Chunk offsets are one scalar
    int64 product plus an int32 tile vector, and a whole chunk skips the
    mask entirely, so only the tail chunk is masked.  The AMSGrad maximum
    store is issued after the parameter and moment stores: measured -30%
    (f32 ams1 333->232us) against the public order; the stored values are
    identical, and the reorder only matters if a caller aliases
    ``max_exp_avg_sqs`` with a main operand, which the public API does not
    support either.
    """
    if HAS_PLANS:
        slot = Meta + 7 * NT + 3 * tl.program_id(0)
        index = tl.load(slot)
        chunk = tl.load(slot + 1)
        stride = tl.load(slot + 2)
    else:
        index = tl.program_id(0)
        chunk = tl.program_id(1)
        stride = tl.num_programs(1)
    Param = tl.load(Meta + index).to(tl.pointer_type(PARAM_DT))
    Grad = tl.load(Meta + NT + index).to(tl.pointer_type(PARAM_DT))
    ExpAvg = tl.load(Meta + 2 * NT + index).to(tl.pointer_type(PARAM_DT))
    ExpAvgSq = tl.load(Meta + 3 * NT + index).to(tl.pointer_type(PARAM_DT))
    MaxExpAvgSq = tl.load(Meta + 4 * NT + index).to(tl.pointer_type(PARAM_DT))
    Step = tl.load(Meta + 5 * NT + index).to(tl.pointer_type(tl.float32))
    n = tl.load(Meta + 6 * NT + index)
    if TENSOR_LR:
        lr_value = tl.load(Lr).to(tl.float32)
    else:
        lr_value = lr.to(tl.float32)
    b1 = beta1.to(tl.float32)
    b2 = beta2.to(tl.float32)
    c1 = one_minus_beta1.to(tl.float32)
    c2 = one_minus_beta2.to(tl.float32)
    decay = weight_decay.to(tl.float32)
    epsilon = eps.to(tl.float32)
    step = tl.load(Step).to(tl.float32)
    bc1 = _adamw_correction(beta1, log_beta1, step, False)
    bc2 = _adamw_correction(beta2, log_beta2, step, False)
    offs = tl.arange(0, BLOCK_SIZE)
    # Full chunks run branch-free in the persistent loop; the one partial
    # tail chunk (if this program owns it) is handled once after the loop.
    # Duplicating the body per chunk measured 345us against 232us here on
    # (2**24,) f32 amsgrad: the second register-pressure tier is real.
    while chunk.to(tl.int64) * BLOCK_SIZE + BLOCK_SIZE <= n:
        offset = chunk.to(tl.int64) * BLOCK_SIZE
        p = tl.load(Param + offset + offs).to(tl.float32)
        g = tl.load(Grad + offset + offs).to(tl.float32)
        m = tl.load(ExpAvg + offset + offs).to(tl.float32)
        v = tl.load(ExpAvgSq + offset + offs).to(tl.float32)
        if MAXIMIZE:
            g = -g
        p = p * (1.0 - lr_value * decay)
        m = b1 * m + c1 * g
        v = b2 * v + c2 * g * g
        variance = v
        if AMSGRAD:
            maximum = tl.load(MaxExpAvgSq + offset + offs).to(tl.float32)
            variance = tl.where(maximum < v, v, maximum)
        denom = tl.sqrt(variance) / tl.sqrt(bc2) + epsilon
        p = p - (lr_value / bc1) * m / denom
        tl.store(Param + offset + offs, p)
        tl.store(ExpAvg + offset + offs, m)
        tl.store(ExpAvgSq + offset + offs, v)
        if AMSGRAD:
            tl.store(MaxExpAvgSq + offset + offs, variance)
        chunk += stride
    offset = chunk.to(tl.int64) * BLOCK_SIZE
    if offset < n:
        mask = offs < (n - offset).to(tl.int32)
        p = tl.load(Param + offset + offs, mask, other=0).to(tl.float32)
        g = tl.load(Grad + offset + offs, mask, other=0).to(tl.float32)
        m = tl.load(ExpAvg + offset + offs, mask, other=0).to(tl.float32)
        v = tl.load(ExpAvgSq + offset + offs, mask, other=0).to(tl.float32)
        if MAXIMIZE:
            g = -g
        p = p * (1.0 - lr_value * decay)
        m = b1 * m + c1 * g
        v = b2 * v + c2 * g * g
        variance = v
        if AMSGRAD:
            maximum = tl.load(MaxExpAvgSq + offset + offs, mask, other=0).to(tl.float32)
            variance = tl.where(maximum < v, v, maximum)
        denom = tl.sqrt(variance) / tl.sqrt(bc2) + epsilon
        p = p - (lr_value / bc1) * m / denom
        tl.store(Param + offset + offs, p, mask)
        tl.store(ExpAvg + offset + offs, m, mask)
        tl.store(ExpAvgSq + offset + offs, v, mask)
        if AMSGRAD:
            tl.store(MaxExpAvgSq + offset + offs, variance, mask)


def _hyperparameters_ok(lr, beta1, beta2, weight_decay, eps):
    """Public-range validation; illegal values fall back to its errors."""
    for name, value in (
        ("beta1", beta1),
        ("beta2", beta2),
        ("weight_decay", weight_decay),
        ("eps", eps),
    ):
        if not isinstance(value, numbers.Real) or not math.isfinite(value):
            return False
        if value < 0 or (name in ("beta1", "beta2") and value >= 1):
            return False
    if isinstance(lr, torch.Tensor):
        # CPU scalar lr is materialized by the public implementation.
        return lr.device.type != "cpu"
    return isinstance(lr, numbers.Real) and math.isfinite(lr) and lr >= 0


def _fast_plan(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
):
    """Validate a whole list with bulk C-level passes and build the meta row.

    Returns ``(dtype, device, values, numels)`` for a launchable list, or
    ``None`` when any check rejects: the public implementation then produces
    its own diagnostics or handles the input itself.  Nothing launches before
    every tensor has been checked, so an invalid tail never updates earlier
    tensors.
    """
    count = len(params)
    if count == 0 or count > _GCU400_MAX_LIST:
        return None
    if (
        len(grads) != count
        or len(exp_avgs) != count
        or len(exp_avg_sqs) != count
        or len(state_steps) != count
        or len(max_exp_avg_sqs) != (count if amsgrad else 0)
    ):
        return None
    if not _hyperparameters_ok(lr, beta1, beta2, weight_decay, eps):
        return None
    first = params[0]
    if not isinstance(first, torch.Tensor):
        return None
    dtype = first.dtype
    if dtype not in _FAST_DTYPES:
        return None
    device = first.device
    # The pointer/numel table needs device-side int64; torch_gcu only turns
    # it on via TORCH_GCU_ENABLE_INT64_AND_UINT64 (the static capability
    # flag does not track that runtime switch, and an int64 upload without
    # it aborts the process).  Gate on the environment directly so a machine
    # without it keeps the public int64-free pointer-only path.
    if os.environ.get("TORCH_GCU_ENABLE_INT64_AND_UINT64") != "1":
        return None
    if not runtime.device.support_int64:
        return None
    if _STREAM_CAPTURING is not None and _STREAM_CAPTURING():
        # The first call uploads the meta table; graph capture stays on the
        # public pointer-only path.
        return None
    # Bulk passes over every operand: dtype and device must be globally
    # uniform (the kernel casts one element type), while shapes only need to
    # match parameter-wise -- heterogeneous groups are the normal case.
    main = params + grads + exp_avgs + exp_avg_sqs
    if amsgrad:
        main += max_exp_avg_sqs
    try:
        if set(map(_MAIN_DESC, main)) != {(dtype, device)}:
            return None
        shapes = tuple(map(_SHAPE, params))
        groups = (grads, exp_avgs, exp_avg_sqs)
        if amsgrad:
            groups += (max_exp_avg_sqs,)
        for group in groups:
            if tuple(map(_SHAPE, group)) != shapes:
                return None
    except AttributeError:
        # A non-tensor entry: the public path raises the real diagnostic.
        return None
    if not all(map(_IS_CONTIGUOUS, main)):
        return None
    try:
        steps_ok = set(map(_STEP_DESC, state_steps)) == {
            (torch.float32, device, torch.strided)
        } and set(map(_NUMEL, state_steps)) == {1}
    except AttributeError:
        return None
    if not steps_ok:
        return None
    if isinstance(lr, torch.Tensor):
        if (
            lr.dtype != torch.float32
            or lr.device != device
            or lr.layout != torch.strided
            or lr.numel() != 1
        ):
            return None
    numels = list(map(_NUMEL, params))
    if all(numels):
        # No filtering; the amsgrad-off kernel never reads the fifth pointer
        # row, so it aliases the second-moment row instead of re-reading it.
        rows = [
            list(map(_DATA_PTR, params)),
            list(map(_DATA_PTR, grads)),
            list(map(_DATA_PTR, exp_avgs)),
            list(map(_DATA_PTR, exp_avg_sqs)),
        ]
        rows.append(list(map(_DATA_PTR, max_exp_avg_sqs)) if amsgrad else rows[3])
        rows.append(list(map(_DATA_PTR, state_steps)))
        values = [pointer for row in rows for pointer in row]
        values.extend(numels)
    else:
        active = [index for index, numel in enumerate(numels) if numel]
        if not active:
            return None
        pick = operator.itemgetter(*active) if len(active) > 1 else None
        rows = []
        for group in (
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs if amsgrad else exp_avg_sqs,
            state_steps,
        ):
            pointers = list(map(_DATA_PTR, group))
            rows.append(pick(pointers) if pick else (pointers[active[0]],))
        values = [value for row in rows for value in row]
        values.extend(numels[index] for index in active)
        numels = [numels[index] for index in active]
    return dtype, device, values, numels


def _gcu400_run(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale,
    found_inf,
):
    """GCU400 launch strategy: one persistent meta-table launch per list."""
    plan = None
    if _GCU400_FAST_PATH and grad_scale is None and found_inf is None:
        plan = _fast_plan(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            amsgrad,
        )
    if plan is None:
        return _ORIGINAL_RUN(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            max_exp_avg_sqs,
            state_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            grad_scale=grad_scale,
            found_inf=found_inf,
        )
    dtype, device, values, numels = plan
    tensor_lr = isinstance(lr, torch.Tensor)
    scalars = (
        0.0 if tensor_lr else float(lr),
        float(beta1),
        float(beta2),
        1.0 - beta1,
        1.0 - beta2,
        math.log(beta1) if beta1 else -math.inf,
        math.log(beta2) if beta2 else -math.inf,
        float(weight_decay),
        float(eps),
    )
    count = len(numels)
    max_numel = max(numels)
    block_size = min(_GCU400_BLOCK_CAP, max(128, 1 << (max_numel - 1).bit_length()))
    chunk_counts = [-(-numel // block_size) for numel in numels]
    has_plans = min(chunk_counts) != max(chunk_counts)
    if has_plans:
        # Uneven tensor sizes: give every tensor programs in proportion to
        # its chunk count so one wave covers the whole list evenly, and let
        # each program stride its own tensor.  Oversized plan tables fall
        # back to the uniform per-tensor grid.
        total_chunks = sum(chunk_counts)
        rows = []
        for tensor, chunks in enumerate(chunk_counts):
            programs = max(
                1, min(chunks, round(_GCU400_PLAN_TARGET * chunks / total_chunks))
            )
            for slot in range(programs):
                rows.extend((tensor, slot, programs))
        if len(rows) <= 3 * _GCU400_MAX_LIST:
            values.extend(rows)
            grid = (len(rows) // 3,)
        else:
            has_plans = False
    if not has_plans:
        grid = (count, min(_GCU400_PERSISTENT, max(chunk_counts)))
    meta = _device_meta(values, device)
    constexprs = (
        tensor_lr,
        amsgrad,
        maximize,
        has_plans,
        block_size,
        count,
        tl_dtype(dtype),
    )
    _launch(
        _adamw_meta_kernel,
        grid,
        constexprs,
        (meta, lr if tensor_lr else None, *scalars, *constexprs),
        (meta, lr) if tensor_lr else (meta,),
        device,
    )
    return None


# The public functions resolve ``_fused_adamw_run`` through module globals;
# the replacement installs the GCU400 strategy while ATen dispatch tracing
# and the tests' monkeypatch of that global keep working.
_IMPL._fused_adamw_run = _gcu400_run

_fused_adamw_ = _IMPL._fused_adamw_
_fused_adamw__tensor_lr = _IMPL._fused_adamw__tensor_lr

__all__ = ["_fused_adamw_", "_fused_adamw__tensor_lr"]
