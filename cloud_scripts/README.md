# 云平台实验操作指南

## 前置条件

- GPU 实例：≥16GB VRAM, CUDA 12.1 兼容
- 已安装 conda + COLMAP
- 数据集已就位（LLFF 和 Mip-NeRF360）

## 目录结构（云实例上）

```
/opt/Sparse_View/
├── DropGaussian_release/    # DropGaussian 代码
├── DDGS/                    # DDGS 代码
├── dataset/
│   ├── nerf_llff_data/      # LLFF 8 个场景
│   └── mipnerf360/          # Mip-NeRF360 7 个场景
└── cloud_scripts/           # 本目录，上传到云实例
```

## 执行流程

### Step 0: 上传

将 `cloud_scripts/` 整个目录上传到云实例 `/root/` 或 `/home/` 下。

### Step 1: 环境搭建（约 10 分钟）

```bash
bash cloud_scripts/setup_env.sh
```

### Step 2: 检查数据集是否需要 COLMAP 预处理

```bash
ls dataset/nerf_llff_data/fern/sparse/ 2>/dev/null && echo "已预处理" || echo "需要预处理"
```

如果显示"需要预处理"，运行 COLMAP（每场景约 10-30min）：

```bash
cd /opt/Sparse_View/DropGaussian_release
# 修改 tools/colmap_llff.py 底部路径后运行
python tools/colmap_llff.py
python tools/colmap_360.py
```

> ⚠️ DropGaussian 和 DDGS 使用**完全相同的 COLMAP 预处理数据**，只需预处理一次。

### Step 3: 应用自适应 Dropout 补丁

```bash
python cloud_scripts/apply_adaptive_dropout.py /opt/Sparse_View/DropGaussian_release
```

这会：
1. 在 `scene/gaussian_model.py` 添加 `compute_importance_scores()` 方法
2. 在 `gaussian_renderer/__init__.py` 替换盲 dropout 为自适应 dropout
3. 自动备份原文件为 `.bak`

### Step 4: 快速验证

```bash
# 单场景快速验证（约 10min）
bash cloud_scripts/run_ablation.sh fern
```

### Step 5: 完整消融实验（LLFF 6 场景 × 4 配置 = 24 runs）

```bash
bash cloud_scripts/run_ablation_llff.sh
```

### Step 6: 结果对比

```bash
python cloud_scripts/compare_results.py \
    --dropgaussian /opt/Sparse_View/DropGaussian_release/output/ablation/baseline \
    --ddgs /opt/Sparse_View/DropGaussian_release/output/ablation/combined \
    --output ablation_comparison.md
```

## 消融实验矩阵

| 实验 | ALPHA_GRAD | 梯度信号 | 密度信号 | 回答什么问题 |
|------|-----------|---------|---------|-------------|
| **baseline** | — | ✗ | ✗ | 原版盲 dropout 基线 |
| **grad_only** | 1.0 | ✓ | ✗ | 梯度引导单独贡献 |
| **density_only** | 0.0 | ✗ | ✓ | 密度引导单独贡献 |
| **combined** | 0.5 | ✓ | ✓ | 两信号互补效果 |

## 代码改动速查

**原版（盲 dropout）**：
```python
# gaussian_renderer/__init__.py
if is_train:
    compensation = torch.ones(opacity.shape[0], device="cuda")
    drop_rate = 0.2 * (iteration/10000)
    d = torch.nn.Dropout(p=drop_rate)
    compensation = d(compensation)
    opacity = opacity * compensation[:, None]
```

**新版（自适应 dropout）**：
```python
if is_train:
    base_drop_rate = 0.2 * (iteration / 10000)
    importance = pc.compute_importance_scores()  # 每个高斯的重要度
    per_gaussian_drop_rate = base_drop_rate * (1.0 - importance)
    keep_prob = 1.0 - per_gaussian_drop_rate
    mask = torch.bernoulli(keep_prob)
    compensation = mask / keep_prob
    opacity = opacity * compensation[:, None]
```

**重要度计算**（`scene/gaussian_model.py` 新增方法）：
```python
def compute_importance_scores(self):
    # 梯度活跃度：对 loss 贡献大的高斯 → 重要
    grad_active = self.xyz_gradient_accum / (self.denom + 1e-8)

    # 局部密度：最近邻距离 → 密集区域的高斯 → 冗余
    nn_dist = distCUDA2(self.get_xyz)
    density = 1.0 / nn_dist

    # 联合：alpha 控制梯度 vs 密度的权重
    alpha = float(os.environ.get("ALPHA_GRAD", 0.5))
    importance = alpha * grad_active + (1-alpha) * (1-density)
    return importance
```

## 训练命令速查

```bash
# 原版 DropGaussian（baseline）
python train.py -s <dataset>/<scene>/ -m <output>/<scene> --eval -r 8 --n_views 3

# 自适应 Dropout（grad+density combined）
ALPHA_GRAD=0.5 python train.py -s <dataset>/<scene>/ -m <output>/<scene> --eval -r 8 --n_views 3

# 仅梯度引导
ALPHA_GRAD=1.0 python train.py -s <dataset>/<scene>/ -m <output>/<scene> --eval -r 8 --n_views 3

# 仅密度引导
ALPHA_GRAD=0.0 python train.py -s <dataset>/<scene>/ -m <output>/<scene> --eval -r 8 --n_views 3
```

## 恢复原版代码

```bash
cd /opt/Sparse_View/DropGaussian_release
mv scene/gaussian_model.py.bak scene/gaussian_model.py
mv gaussian_renderer/__init__.py.bak gaussian_renderer/__init__.py
```
