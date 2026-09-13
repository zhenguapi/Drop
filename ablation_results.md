# DropGaussian 自适应 Dropout 消融实验记录

> 最后更新: 2026-07-19
> 基线: DropGaussian (CVPR 2025), 无先验
> 数据集: LLFF 3-view

---

## v1 消融实验 (30000 iter, 6 场景)

> 【2026-09-13 T0 决议② 更正标注】本节标题"6 场景"与下文"v1 vs v2 关键差异"表中"场景数 6 (无 fortress, room)"
> 均为陈旧标注，作废。经 `train_changed.log` 核对：v1 实际为 **4 配置 × 8 场景 = 32 runs**（含 fortress/room，
> run 头 `[1/32]`–`[32/32]`，两者均有完整 PSNR/SSIM/LPIPS 指标行），逐场景 Δ 与日志重算一致（±0.002）。
> **以逐场景表（8 行）为准。** 决议详情：本地 `docs/tasks/T0_governance/output.md`。

**方法**: 位置梯度 + 几何密度 (distCUDA2)

**公式**: `importance = α × grad_active + (1-α) × (1 - density)`

| 实验 | α | PSNR | Δ vs Baseline | SSIM | LPIPS |
|------|---|------|--------------|------|-------|
| Baseline (盲 dropout) | — | 20.213 | — | 0.704 | 0.209 |
| Grad-only | 1.0 | 20.091 | -0.12 | 0.689 | 0.239 |
| Density-only | 0.0 | 19.403 | -0.81 | 0.662 | 0.225 |
| **Combined** | **0.5** | **20.357** | **+0.14** | 0.705 | 0.208 |

**逐场景 Combined vs Baseline**:

| 场景 | ΔPSNR | 定性 |
|------|-------|------|
| trex | +0.71 | 大幅提升 |
| room | +0.47 | 显著提升 |
| fortress | +0.31 | 提升 |
| leaves | +0.26 | 提升 |
| fern | -0.01 | 持平 |
| orchids | -0.03 | 持平 |
| flower | -0.25 | 下降 |
| horns | -0.30 | 下降 |

**核心发现**:
1. 密度信号单独用有害 (-0.81 dB)
2. 梯度+密度互补才有效
3. 场景差异大——物体边界清晰的场景 (trex) 效果最好

---

## v2 消融实验 (10000 iter, 8 场景)

**方法**: 位置梯度 + opacity 梯度 + 几何密度

**公式**: `importance = α × grad_active + β × op_grad + γ × (1 - density)`

| 实验 | α | β | γ | 说明 |
|------|---|---|---|------|
| baseline | 0 | 0 | 0 | 均匀 dropout (≈原始盲 dropout) |
| opgrad_only | 0 | 1.0 | 0 | 纯 opacity 梯度 |
| opgrad_grad | 0.5 | 0.5 | 0 | opacity梯度 + 位置梯度 |
| all_three | 0.3 | 0.5 | 0.2 | 三信号联合 |

### 逐场景 PSNR

| Scene | baseline | opgrad_only | opgrad_grad | all_three |
|-------|----------|-------------|-------------|-----------|
| fern | **22.85** | 22.68 | 22.79 | 22.77 |
| flower | **20.88** | 20.69 | 20.41 | 20.63 |
| fortress | 22.91 | 22.83 | **23.05** | 22.18 |
| horns | 19.13 | 19.08 | **19.36** | 18.99 |
| leaves | 17.46 | **17.73** | 17.55 | 17.34 |
| orchids | 15.89 | **15.98** | 15.86 | 15.86 |
| room | **22.23** | 21.59 | 21.99 | 22.21 |
| trex | **22.68** | 22.21 | 22.61 | 22.60 |
| **Avg** | **20.50** | 20.35 | 20.45 | 20.32 |

### 汇总

| 指标 | opgrad_only | opgrad_grad | all_three |
|------|-------------|-------------|-----------|
| Δ vs baseline | **-0.15** | **-0.05** | **-0.18** |
| 胜/平/负 | 2/0/6 | 3/1/4 | 0/2/6 |

**结论**: opacity 梯度信号在 10000 iter 下无提升，三个 v2 变体全部低于 baseline。

---

## v1 vs v2 关键差异

| 维度 | v1 | v2 |
|------|-----|-----|
| 信号 | 位置梯度 + distCUDA2 密度 | 位置梯度 + opacity梯度 + distCUDA2 密度 |
| 信号产生时机 | distCUDA2 从初始化就有值 | opacity 梯度需要 loss.backward() 后才有 |
| 早期有效性 | 第 1 轮就有效 | EMA 冷启动，前数十轮信号为零 |
| 迭代数 | 30000 | 10000 |
| 归一化 | max 归一化 | 同 |
| 场景数 | 6 (无 fortress, room) | 8 (全) |

---

## v2 代码改动位置

```
DropGaussian_release/
├── scene/gaussian_model.py
│   ├── __init__         → +self.opacity_grad_cache
│   ├── capture/restore  → +opacity_grad_cache 序列化
│   ├── training_setup   → 初始化 cache 为 zeros
│   ├── prune_points     → 同步裁剪 cache
│   ├── densification_postfix → reset cache for new Gaussians
│   ├── cache_opacity_gradient()  → EMA 缓存 |∂L/∂α| (NEW)
│   └── compute_importance_scores() → 三信号公式 (REWRITTEN)
├── gaussian_renderer/__init__.py → 自适应 dropout (调用 compute_importance_scores)
└── train.py → loss.backward() 后插入 cache_opacity_gradient()
```

## 实验脚本

```
cloud_scripts/
├── setup_env.sh              # 环境搭建
├── run_dropgaussian.sh       # 完整复现 (10000 iter)
├── run_ablation_llff.sh      # v2 消融实验 (已修复 metrics 文件路径)
└── README.md                 # v2 文档
```

## 当前状态

v2 (opacity 梯度) 实验结论为负面。已回到苏格拉底式讨论，重新审视：
1. v1 密度信号 (distCUDA2) 为什么有效？
2. 如何改进密度信号而不引入外部先验？
3. 如何区分"必要高密度"和"冗余高密度"？

当前讨论点: distCUDA2 从初始化就有值，不依赖训练进度——这是它相比 opacity 梯度的关键优势。
