"""cummaxmin_backward 的 GCU400 特化实现（scatter-add 语义）。

语义：``grad_input.scatter_add_(dim, indices, grad_output)``，重复 index 累加，
累加在 float32 完成后再 cast 回输入 dtype。

GCU 上的两条实测结论决定了本实现的形态：
- ``torch.zeros`` 的 memset 只有约 2 GB/s（4096x4096 的 67MB 需要约 31ms），
  而 Triton 内核写同样大小缓冲约 0.1ms，因此清零统一用 ``_zero_kernel``；
- ``tl.atomic_add`` 在本硬件上非常慢（4096x4096 约 30ms，是同规模普通
  store 的 100 倍），因此大归约轴改用"前缀和 + 段边界"的无原子路径。

无原子路径（``_cmb_prefix_kernel``）的推导：当 ``indices`` 沿归约轴非递减时，
每个目标位置 t 对应归约轴上一段连续区间 [s(t), e(t)]，于是
``out[t] = S[e(t)] - S[s(t)-1]``，其中 S 为归约轴上的前缀和。段首/段尾的位置
可以直接用相邻 indices 比较得到，且每个 t 只会被写一次（无写冲突）：
- 段首处把 ``S[r]-g[r]`` 写入 start 缓冲；
- 段尾处把 ``S[r]`` 写入输出；
- 收尾 ``out -= start`` 一次元素级减法即得结果。
若某行 indices 非递减不成立，该行在同一内核内回退到原子路径，语义仍正确。
"""

import torch
import triton
import triton.language as tl

__all__ = ["cummaxmin_backward"]

# Upper limit of grid.x on GCU
_MAX_GRID_X = 65535
# Best BLOCK for elementwise kernels (zeroing / final subtraction): measured 32768 takes only 0.11ms for 268MB,
# clearly better than 1024 (3.4ms) and 8192 (0.43ms), so these two kernels use a large BLOCK to fill a single wave;
# For small buffers, round numel up to a power of 2 to avoid using only a few of the 32768 lanes.
_ELEM_BLOCK = 32768
# Dual thresholds for enabling the atomic-free "prefix sum + segment boundaries" path (from an empirical sweep):
# Atomic overhead only becomes the dominant cost when the reduction axis is long enough and there are enough total
# elements; on small shapes, the atomic path
# is faster because it avoids one indices read and one kernel. From RS=1024/numel=1M, the prefix path leads by 6.8x,
# and the two are on par at RS=512/numel=128K, so 512 + 256K is used as the switch point.
_PREFIX_RS_THRESHOLD = 512
_PREFIX_NUMEL_THRESHOLD = 262144
# Threshold for using torch.zeros for small buffers: measured 64x64 (4096 elements) with torch.zeros is actually slower
# than
# torch.empty + a Triton zeroing kernel by 20% (0.093ms vs 0.076ms), so always use the Triton kernel.
_SMALL_NUMEL = 0


@triton.jit
def _zero_kernel(OUT, N, BLOCK: tl.constexpr):
    """把 OUT 的前 N 个元素清零（grid-stride，避免 grid.x 溢出）。"""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        tl.store(OUT + pos, 0.0, mask=pos < N)


@triton.jit
def _sub_kernel(OUT, SUB, N, BLOCK: tl.constexpr):
    """OUT -= SUB（元素级，grid-strided，原地）。"""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        m = pos < N
        tl.store(
            OUT + pos, tl.load(OUT + pos, mask=m) - tl.load(SUB + pos, mask=m), mask=m
        )


@triton.jit
def _sub_cast_kernel(OUT, SUB, DST, N, BLOCK: tl.constexpr):
    """DST = OUT - SUB，并直接写成输入 dtype：把"收尾减法"和"回写 dtype"合成一次访存。"""
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0) * BLOCK
    for bs in tl.range(tl.program_id(0) * BLOCK, N, step):
        pos = bs + ar
        m = pos < N
        val = tl.load(OUT + pos, mask=m) - tl.load(SUB + pos, mask=m)
        tl.store(DST + pos, val.to(DST.dtype.element_ty), mask=m)


@triton.jit
def _cmb_kernel(G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr):
    """原子路径（归约轴在最内层）。

    grid.x 上限 65535，因此用固定 grid + grid-stride 遍历 outer 行。
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // IS
        inner = row % IS
        base = outer * (RS * IS) + inner
        for bs in range(0, RS, BLOCK):
            pos = bs + ar
            if MASKED:
                m = pos < RS
                g = tl.load(G + base + pos * IS, mask=m, other=0.0)
                t = tl.load(IDX + base + pos * IS, mask=m, other=0)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + base + pos * IS)
                t = tl.load(IDX + base + pos * IS)
                tl.atomic_add(OUT + base + t * IS, g.to(tl.float32))


@triton.jit
def _cmb_inner_kernel(
    G, IDX, OUT, NROW, RS, IS, BLOCK: tl.constexpr, MASKED: tl.constexpr
):
    """原子路径（归约轴在中间）：沿连续 inner 轴向量化。

    ``row`` 索引 (outer, r)；每个 lane scatter ``grad[outer, r, j]`` 到
    ``grad_input[outer, t, j]``，``t`` 来自前向 indices，inner 列 j 保持不变。
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        outer = row // RS
        outer_base = outer * (RS * IS)
        src_base = outer_base + (row % RS) * IS
        for bs in range(0, IS, BLOCK):
            j = bs + ar
            if MASKED:
                m = j < IS
                g = tl.load(G + src_base + j, mask=m, other=0.0)
                t = tl.load(IDX + src_base + j, mask=m, other=0)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32), mask=m)
            else:
                g = tl.load(G + src_base + j)
                t = tl.load(IDX + src_base + j)
                tl.atomic_add(OUT + outer_base + t * IS + j, g.to(tl.float32))


@triton.jit
def _cmb_prefix_kernel(G, IDX, OUT, START, NROW, RS, BLOCK: tl.constexpr):
    """无原子路径（归约轴在最内层，indices 沿归约轴非递减时成立）。

    每个 program 处理一整行：先验证该行 indices 非递减，成立则走前缀和 +
    段边界写，否则本行回退到原子 scatter。每行的 t 在无原子路径中最多被
    写一次，因此不需要原子。
    """
    ar = tl.arange(0, BLOCK)
    step = tl.num_programs(0)
    for row in tl.range(tl.program_id(0), NROW, step):
        base = row * RS

        # First pass: verify that indices are non-decreasing (no comparison at pos==0)
        bad = 0
        for bs in range(0, RS, BLOCK):
            pos = bs + ar
            m = pos < RS
            cur = tl.load(IDX + base + pos, mask=m, other=0)
            prv = tl.load(IDX + base + pos - 1, mask=(pos > 0) & m, other=0)
            bad += tl.sum(((cur < prv) & (pos > 0) & m).to(tl.int32), 0)

        if bad == 0:
            # Second pass: prefix sum + segment-start/segment-end writes (no write conflicts)
            carry = 0.0
            for bs in range(0, RS, BLOCK):
                pos = bs + ar
                m = pos < RS
                g = tl.load(G + base + pos, mask=m, other=0.0).to(tl.float32)
                cur = tl.load(IDX + base + pos, mask=m, other=0)
                prv = tl.load(IDX + base + pos - 1, mask=(pos > 0) & m, other=0)
                nxt = tl.load(IDX + base + pos + 1, mask=(pos + 1 < RS) & m, other=0)
                t = cur.to(tl.int32)
                s = tl.cumsum(g, 0) + carry
                is_start = (pos == 0) | (cur != prv)
                is_end = (pos == RS - 1) | (cur != nxt)
                tl.store(START + base + t, s - g, mask=m & is_start)
                tl.store(OUT + base + t, s, mask=m & is_end)
                carry += tl.sum(g, 0)
        else:
            # Non-decreasing check failed: fall back to the atomic path for this row
            for bs in range(0, RS, BLOCK):
                pos = bs + ar
                m = pos < RS
                g = tl.load(G + base + pos, mask=m, other=0.0)
                t = tl.load(IDX + base + pos, mask=m, other=0)
                tl.atomic_add(OUT + base + t, g.to(tl.float32), mask=m)


def _zero_fill(numel, dtype, device):
    """分配并清零一维缓冲。

    小缓冲上 ``torch.zeros`` 的 memset 与 Triton 内核相当，但少一次内核发射，
    因此 size 不超过 ``_SMALL_NUMEL`` 时直接用 ``torch.zeros``。
    """
    if numel <= _SMALL_NUMEL:
        return torch.zeros(numel, dtype=dtype, device=device)
    buf = torch.empty(numel, dtype=dtype, device=device)
    BLOCK = min(_ELEM_BLOCK, max(1024, triton.next_power_of_2(numel)))
    grid = (min(triton.cdiv(numel, BLOCK), _MAX_GRID_X),)
    _zero_kernel[grid](buf, numel, BLOCK=BLOCK, num_warps=8)
    return buf


def cummaxmin_backward(grad_output, input, indices, dim):
    ndim = grad_output.ndim
    if dim < 0:
        dim += ndim
    shape = list(grad_output.shape)
    RS = shape[dim]
    inner_size = 1
    for i in range(dim + 1, ndim):
        inner_size *= shape[i]
    outer_size = 1
    for i in range(dim):
        outer_size *= shape[i]

    if RS == 0:
        return torch.zeros(shape, dtype=grad_output.dtype, device=grad_output.device)

    # Make non-contiguous inputs contiguous first: all kernels address data in [outer, reduction, inner] row-major order
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if not indices.is_contiguous():
        indices = indices.contiguous()

    numel = outer_size * RS * inner_size
    use_prefix = (
        inner_size == 1
        and RS >= _PREFIX_RS_THRESHOLD
        and numel >= _PREFIX_NUMEL_THRESHOLD
    )
    if use_prefix:
        # The output and segment-start buffer share one allocation; zeroing runs only one kernel
        buf = _zero_fill(2 * numel, torch.float32, grad_output.device)
        out = buf[:numel].view(shape)
        start = buf[numel:]
        _cmb_prefix_kernel[(min(outer_size, _MAX_GRID_X),)](
            grad_output,
            indices,
            out,
            start,
            outer_size,
            RS,
            BLOCK=min(triton.next_power_of_2(RS), 2048),
            num_warps=8,
        )
        BLOCK = min(_ELEM_BLOCK, max(1024, triton.next_power_of_2(numel)))
        grid = (min(triton.cdiv(numel, BLOCK), _MAX_GRID_X),)
        if grad_output.dtype is torch.float32:
            _sub_kernel[grid](out, start, numel, BLOCK=BLOCK, num_warps=8)
        else:
            # Subtract and write back to the target dtype in one pass: avoids one fp32 write and one cast kernel
            dst = torch.empty(shape, dtype=grad_output.dtype, device=out.device)
            _sub_cast_kernel[grid](out, start, dst, numel, BLOCK=BLOCK, num_warps=8)
            return dst
    else:
        out = _zero_fill(numel, torch.float32, grad_output.device).view(shape)
        if inner_size == 1:
            BLOCK = min(triton.next_power_of_2(RS), 1024)
            _cmb_kernel[(min(outer_size, _MAX_GRID_X),)](
                grad_output,
                indices,
                out,
                outer_size,
                RS,
                inner_size,
                BLOCK=BLOCK,
                MASKED=(BLOCK != RS),
                num_warps=8,
            )
        else:
            BLOCK = min(triton.next_power_of_2(inner_size), 1024)
            nrow = outer_size * RS
            _cmb_inner_kernel[(min(nrow, _MAX_GRID_X),)](
                grad_output,
                indices,
                out,
                nrow,
                RS,
                inner_size,
                BLOCK=BLOCK,
                MASKED=(BLOCK != inner_size),
                num_warps=8,
            )
    if grad_output.dtype is torch.float32:
        return out
    return out.to(grad_output.dtype)
