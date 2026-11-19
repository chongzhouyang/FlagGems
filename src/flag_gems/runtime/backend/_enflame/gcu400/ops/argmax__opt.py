"""argmax 入口（保留 `__opt` 模块名，由 ops/__init__.py 导入）。

历史：本文件原本是"不改 argmax.py、只显式传 constexpr 直调 kernel"的调优候选
（inner：G48/BM8/TILE_N=npo2(N)/nw4；non_inner：TILE_K=512/TILE_N=64/nw4）。

2026-11-17 起两条 kernel 因正确性缺陷被重写（见 argmax.py 内注释）：
  * 2D tile 上 `tl.max(return_indices=True)` 归约在 K 需要多个 tile 时错位；
  * 固定 48 CTA 的 grid-stride 让错误率接近 100%；
  * 计算得到的 int64 下标向量会按 int32 步长落盘。
重写后 kernel 是 1D 行循环 + int32 下标，签名不再有
TILE_N/BLOCK_M/GRID_SIZE/ONE_TILE_PER_CTA，本模块的显式直调随之失效。
调优旋钮已收敛到 argmax.py 的 `_inner_config` / `_non_inner_config`
（可用 FG_ARGMAX_INNER_TILE / FG_ARGMAX_NON_INNER_TILE 覆盖），
因此这里直接委托生产实现，避免出现"入口指向已删签名"的 TypeError。
"""

from flag_gems.runtime.backend._enflame.gcu400.ops.argmax import argmax

__all__ = ["argmax"]
