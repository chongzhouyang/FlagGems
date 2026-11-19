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

logger = logging.getLogger(__name__)


@triton.jit
def _fp8_mqa_logits_kernel(
    Q,
    K,
    K_SCALES,
    WEIGHTS,
    CU_SEQLEN_KS,
    CU_SEQLEN_KE,
    LOGITS,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    PAD_M: tl.constexpr,
):
    """GCU400 FP8 MQA logits。

    logits[m, n] = sum_h( ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * w[m,h] )，
    K 每个 n 只搬一次在 H 维复用；CLEAN_LOGITS 为真时同 launch 内把
    [cu_seqlen_ks, cu_seqlen_ke) 之外的位置写成 -inf。

    相对原实现的三处 GCU400 关键改动：
      * M/N 从 `tl.constexpr` 改回运行期标量：否则每个新 (M, N) 都触发一次
        重编译（N=4096 的长上下文 decode 首调延迟很高）。
      * `tl.dot` 的行维必须 >=16 且为 16 的倍数，故行索引统一走 PAD_M 补齐、
        用 BLOCK_M 掩码控制真实行数；原写法在 M<16 时静默算错。
      * HEAD_GROUP 把相邻若干 head 叠到 dot 的行轴（行号 = head*PAD_M + row），
        再 reshape 回 [HEAD_GROUP, PAD_M, BLOCK_N] 沿 head 归约；decode 小 M
        时把 MMA 行维吃满（M=1、N=4096 实测 0.19ms -> 0.080ms），prefill 用
        HEAD_GROUP=1 退回逐 head 循环。
      * 不再对 logits 做 `torch.zeros` 预清零（clean_logits 为假时全量写入）。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stacked = tl.arange(0, HEAD_GROUP * PAD_M)
    head_off = stacked // PAD_M
    row_off = stacked % PAD_M
    offs_m = pid_m * BLOCK_M + row_off
    stack_mask = (row_off < BLOCK_M) & (offs_m < M)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_d = tl.arange(0, D)

    k_scales = tl.load(K_SCALES + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros([PAD_M, BLOCK_N], dtype=tl.float32)
    for h_start in range(0, H, HEAD_GROUP):
        q = tl.load(
            Q
            + offs_m[:, None] * stride_qm
            + (h_start + head_off)[:, None] * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=stack_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        score = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        score = tl.maximum(score * k_scales[None, :], 0.0)

        weight = tl.load(
            WEIGHTS + offs_m * H + (h_start + head_off), mask=stack_mask, other=0.0
        ).to(tl.float32)
        if HEAD_GROUP == 1:
            acc += score * weight[:, None]
        else:
            acc += tl.sum(
                tl.reshape(score * weight[:, None], [HEAD_GROUP, PAD_M, BLOCK_N]),
                axis=0,
            )

    out_rows = pid_m * BLOCK_M + tl.arange(0, PAD_M)
    row_ok = out_rows < M
    if CLEAN_LOGITS:
        ks_start = tl.load(CU_SEQLEN_KS + out_rows, mask=row_ok, other=0)
        ke_end = tl.load(CU_SEQLEN_KE + out_rows, mask=row_ok, other=N)
        n_valid = (offs_n[None, :] >= ks_start[:, None]) & (
            offs_n[None, :] < ke_end[:, None]
        )
        acc = tl.where(n_valid, acc, float("-inf"))

    out_ptrs = LOGITS + out_rows[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=row_ok[:, None] & mask_n[None, :])


def _select_tile(m: int) -> tuple:
    """按行数选 tile（GCU400 实测最优，见 kernel docstring）。

    返回 (BLOCK_M, BLOCK_N, HEAD_GROUP, PAD_M)。
    """
    if m <= 16:
        return 16, 128, 8, 16
    if m <= 64:
        return 16, 256, 4, 16
    return 64, 256, 1, 64


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME FP8_MQA_LOGITS")

    k_fp8, k_scales = kv

    M, H, D = q.shape
    N = k_fp8.shape[0]

    # Both clean_logits branches write the whole tensor (invalid positions set to -inf), so no torch.zeros pre-clear is
    # needed
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    block_m, block_n, head_group, pad_m = _select_tile(M)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q.stride(0),  # stride_qm
        q.stride(1),  # stride_qh
        q.stride(2),  # stride_qd
        k_fp8.stride(0),  # stride_kn
        k_fp8.stride(1),  # stride_kd
        M,
        N,
        H,
        D,
        clean_logits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_GROUP=head_group,
        PAD_M=pad_m,
        num_warps=4,
    )

    return logits
