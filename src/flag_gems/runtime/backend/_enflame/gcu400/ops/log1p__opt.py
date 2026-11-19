"""log1p 候选 v2：按 dtype / 元素量分档的 launch 配置（2026-11-16 性能轮）。

v1（11-02 TRACKED 扫描）把 `BLOCK=32768 / GRID=48 / num_warps=4 /
NUM_STAGES=4` 写死并按 dtype 分化出 fp16/bf16 BS65536 nw2 ST1 与 fp32
BS8192 nw4 ST4 两档。本轮（11-16，GCU400/Libra，官方 bench kernel 模式
同口径 do_bench median w1000/r100）在 40+ 配置 × 6 变体函数体上做了
消融阶梯与扫参，结论：

- **copy 地板**（同形态 flat gs-cap48 掩码 kernel，仅去 log）：fp32 2^24
  48-51us（2.7-2.8TB/s）、2^29 1418us（3.03TB/s）≈ native log1p fp32
  （50.9us / 1422us）——native fp32 已贴 1R1W 地板，该 dtype 的天花板
  即 ~1.0×。
- **fp32 中大形状 0.64-0.68× 的根因是编译器级**：`tl.log` 在 GCU backend
  落到不透明 extern call `tops_log_v1_fp32(vector<128xf32>)`（llir 实
  证），无法与 2× 宽的 fp32 load/store 流水重叠；BS8192/nw4/ST4 已是
  (BS, ns, nw, grid, 函数体) 全空间的最优解（75.1us vs 地板 50.2us）。
  grid 扩张（96/192/dense）、EVEN 免 mask、ns1-8 全档均无收益或劣化。
- **函数体全空间否决**（fp32 2^24，对照 extern tl.log 75.1us）：
  `fast_logf` extern 与 tl.log 逐 config 同价（同链复证）；`fast_log2f·ln2`
  +12%；准确 `__ef_logf` extern 3.0×；`__ef_log1pf` extern 4.3×；
  **纯 Triton 位分解 + D9 极小极大 Horner 多项式（fp32 模拟误差 5.96e-8）
  10.6× 劣化**（801.9us）——backend 对 bitcast/where 多 op 元素链的中间
  值物化病态（消融阶梯：纯位分解体 82.9us、Horner+select 体 119.9us、
  去 select 体 130.9us、加 4 个特殊值 select 后 801.9us），lgamma Lanczos
  纯体先例不适用于含 bitcast/select 的链。编译器问题已按 P 目录格式
  归档（tops_log_v1_fp32 extern call 不可流水）。
- **fp16/bf16 大形状**：n ≥ 2^26 起 BS131072 反超 BS65536（2^29: 785.7
  vs 805.2us，+2.4%；2^26: 104.5 vs 107.0us；2^25 打平），分界取 2^26。
- **小形状**（n ≤ 4096）：BS 减半到 `max(512, npo2(n//2))` 多 CTA 摊
  launch 延迟（4096 元素：fp32 4.0→3.7us、fp16 3.4→3.3us，nw2/ST1）；
  n ≤ 512 以下 grid=1 与更小 BS 打平。

host 侧 dtype promotion / 非连续 / 空张量语义与生产 `log1p()` 逐字同构。
2026-11-16 晚轮起 `_config` 分档表单源化到生产 `log1p.py`（本文件 import 复
用），`log1p_` / `log1p_out` 的 `_launch` 同步套用同一分档——inplace/out
与 functional 共用上述 (BS, grid, nw, ns) 全套证据链；inplace 大形状此前
停在 TRACKED 旧配置 BS16384/nw4/ST4（官方 bench core 15 cell 平均仅
0.70×，fp32 2^29 0.46×），套用后与 functional 同档。

注意探针踩坑：`torch.randn()+2.0` 仍会产出 x<-1 使 log1p 得 NaN，
而 `torch.equal` 对 NaN 恒 False，会让全部配置误判为不合格；
扫描/门禁必须用全正输入。
"""

import torch

from flag_gems.runtime import torch_device_fn
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import _config
from flag_gems.runtime.backend._enflame.gcu400.ops.log1p import (
    log1p_kernel as _kernel_wrapped,
)

_K = _kernel_wrapped.fn

__all__ = ["log1p"]


def log1p(x: torch.Tensor):
    if x.is_floating_point():
        result_dtype = x.dtype
    else:
        result_dtype = torch.float32

    x_contig = x.to(result_dtype).contiguous()
    out = torch.empty_like(x_contig)
    n = out.numel()
    if n == 0:
        return out.view_as(x)

    BS, G, nw, ST = _config(n, result_dtype)
    grid = (G, 1, 1)
    with torch_device_fn.device(out.device.index):
        _K[grid](
            x_contig.view(-1),
            out.view(-1),
            n,
            BLOCK_SIZE=BS,
            GRID_SIZE=G,
            NUM_STAGES=ST,
            num_warps=nw,
        )
    return out.view_as(x)
