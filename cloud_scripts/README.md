# 云平台实验操作指南 (v2)

## v2 核心机制

用 **|∂L/∂α|**（opacity 梯度）度量功能冗余度：

```
如果调一个 Gaussian 的透明度，loss 变化大 → 它的贡献独特不可替代 → 重要 → 少丢
如果调它的透明度，loss 几乎不变        → 别人能替 → 冗余 → 多丢
```

### 三信号 importance 公式

```
importance = α × position_grad + β × opacity_grad + γ × (1 - density)

α = ALPHA_GRAD    位置梯度权重（几何学习信号，默认 0.3）
β = BETA_OPGRAD   opacity 梯度权重（功能冗余度信号，默认 0.5）
γ = GAMMA_DENSITY 几何密度权重（辅助信号，默认 0.2）
```

---

## 前置条件

- GPU 实例：≥16GB VRAM, CUDA 12.1 兼容
- 已安装 conda
- 数据集已就位（LLFF 和 Mip-NeRF360，需 COLMAP 预处理完成）

## 目录结构（云实例上）

```
/opt/Sparse_View/
├── DropGaussian_release/    # DropGaussian 代码（已含 v2 自适应 dropout）
├── DDGS/                    # DDGS 代码（提供 conda 环境和 CUDA 扩展）
├── dataset/
│   ├── nerf_llff_data/      # LLFF 8 个场景
│   └── mipnerf360/          # Mip-NeRF360 7 个场景
└── cloud_scripts/           # 本目录
```

## 执行流程

### Step 1: 环境搭建（约 10 分钟）

```bash
bash cloud_scripts/setup_env.sh
```

### Step 2: 快速验证

```bash
bash cloud_scripts/run_dropgaussian.sh quick
```

### Step 3: 完整消融实验（LLFF 8 场景 × 4 配置 = 32 runs）

```bash
bash cloud_scripts/run_ablation_llff.sh
```

---

## 消融实验矩阵 (v2)

| 实验 | ALPHA_GRAD | BETA_OPGRAD | GAMMA_DENSITY | 回答什么问题 |
|------|-----------|-------------|---------------|-------------|
| **baseline** | 0 | 0 | 0 | 均匀 dropout（≈原始盲 dropout）基线 |
| **opgrad_only** | 0 | 1.0 | 0 | 纯 opacity 梯度单独有效吗？ |
| **opgrad_grad** | 0.5 | 0.5 | 0 | opacity梯度+位置梯度 比 v1 好吗？ |
| **all_three** | 0.3 | 0.5 | 0.2 | 三信号联合（默认配置） |

---

## 训练命令速查

```bash
cd /opt/Sparse_View/DropGaussian_release
conda activate DDGS

# Baseline（均匀 dropout）
ALPHA_GRAD=0 BETA_OPGRAD=0 GAMMA_DENSITY=0 \
  python train.py -s dataset/nerf_llff_data/fern/ -m output/test/fern \
    --eval -r 8 --n_views 3 --iterations 10000

# 纯 opacity 梯度
ALPHA_GRAD=0 BETA_OPGRAD=1.0 GAMMA_DENSITY=0 \
  python train.py -s dataset/nerf_llff_data/fern/ -m output/test/fern \
    --eval -r 8 --n_views 3 --iterations 10000

# opacity梯度 + 位置梯度
ALPHA_GRAD=0.5 BETA_OPGRAD=0.5 GAMMA_DENSITY=0 \
  python train.py -s dataset/nerf_llff_data/fern/ -m output/test/fern \
    --eval -r 8 --n_views 3 --iterations 10000

# 三信号联合（默认）
python train.py -s dataset/nerf_llff_data/fern/ -m output/test/fern \
    --eval -r 8 --n_views 3 --iterations 10000
```

## 渲染 + 评估

```bash
python render.py -m output/test/fern --eval -r 8
python metric.py --path output/test --iteration 10000
```

---

## 代码改动速查

### 新增方法 1: `cache_opacity_gradient()`（scene/gaussian_model.py）

```python
def cache_opacity_gradient(self, ema_decay=0.9):
    """在 loss.backward() 后调用，EMA 缓存 |∂L/∂α|"""
    if self._opacity.grad is not None:
        grad_abs = self._opacity.grad.detach().abs()
        if self.opacity_grad_cache.shape[0] != grad_abs.shape[0]:
            self.opacity_grad_cache = grad_abs.clone()
        else:
            self.opacity_grad_cache = (
                ema_decay * self.opacity_grad_cache + (1.0 - ema_decay) * grad_abs
            )
```

### 新增方法 2: `compute_importance_scores()`（scene/gaussian_model.py）

```python
def compute_importance_scores(self):
    # 信号 1: 位置梯度活跃度
    grad_active = self.xyz_gradient_accum / (self.denom + 1e-8)

    # 信号 2: opacity 梯度 → 功能冗余度
    op_grad = self.opacity_grad_cache.squeeze()

    # 信号 3: 几何密度
    nn_dist = distCUDA2(self.get_xyz)
    density = 1.0 / nn_dist

    # 组合（env var 控制权重）
    alpha = float(os.environ.get("ALPHA_GRAD", 0.3))
    beta  = float(os.environ.get("BETA_OPGRAD", 0.5))
    gamma = float(os.environ.get("GAMMA_DENSITY", 0.2))
    return alpha * grad_active + beta * op_grad + gamma * (1 - density)
```

### 调用时序（train.py）

```python
loss.backward()
gaussians.cache_opacity_gradient()  # ← 缓存 |∂L/∂α|，供下轮 render 使用
# ...
gaussians.optimizer.step()
gaussians.optimizer.zero_grad(set_to_none=True)
```

### 自适应 dropout（gaussian_renderer/__init__.py）

```python
if is_train:
    base_drop_rate = 0.2 * (iteration / 10000)
    importance = pc.compute_importance_scores()
    per_gaussian_drop_rate = base_drop_rate * (1.0 - importance)
    keep_prob = 1.0 - per_gaussian_drop_rate
    mask = torch.bernoulli(keep_prob)
    compensation = mask / keep_prob
    opacity = opacity * compensation[:, None]
```

---

## 如需原始 DropGaussian baseline

v2 改动已直接提交到代码中。如需原始盲 dropout：

```bash
cd /opt/Sparse_View/DropGaussian_release
git stash                    # 暂存 v2 改动
# 或
git checkout <original_commit_hash>
```
