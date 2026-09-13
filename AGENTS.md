# AGENTS.md — 云服务器实验环境上下文

> 面向在本仓库（`DropGaussian_release`）工作的 AI 助手（ZCode CLI / Claude Code 等）或人类操作者。本文件随 git 同步，云端 `git pull` 后即为最新。
> 最后更新：2026-09-13 ｜ 本地完整文档体系在 `/opt/Sparse_View/docs/`（云端不可见，关键上下文以本文件为准）

## 1. 研究背景（30 秒版）

**定位**：本仓库的实验是大论文《基于 3DGS 的三维重建在枳壳苗根系表型性状提取中的应用》创新点一（稀疏视角重建改进）与期刊小论文的**素材线**；大论文优先原则下"量力执行"——仅在云 GPU 空闲时段推进。

稀疏视角（3-view）3DGS 的自适应 dropout 正则：用 importance 信号逐高斯调节 dropout 率（`base_drop_rate = 0.2 × iteration/10000`，Bernoulli 采样 + opacity 补偿保证无偏）。基线 DropGaussian (CVPR 2025)，无深度先验。

当前公式：`importance = α·位置梯度 + β·opacity梯度 + γ·(1−密度)`，权重走 env var。
- v1（α=0.5, γ=0.5，30000 iter）：LLFF 平均 **+0.14 dB**（弱正向）
- v2（+opacity 梯度，10000 iter）：三变体全负 → 引出理论框架与证伪实验（当前阶段）
- 关键机制代码：`scene/gaussian_model.py`（`compute_importance_scores()` ~437 行、权重 env var ~494–501 行）、`gaussian_renderer/__init__.py`（自适应 dropout ~87–107 行）、`train.py`（`loss.backward()` 后调 `cache_opacity_gradient()`，~91 行）

## 2. 环境事实

| 项 | 值 |
|----|-----|
| 代码目录（脚本 `cd` 的位置） | `/root/DropGaussian_release` |
| conda 环境 | `DDGS` |
| PyTorch / Python / CUDA | 2.5.1 / 3.12 / 12.4（autodl 实例，2026-09 由用户确认） |
| GPU | vGPU-32GB × 1 |
| LLFF 数据 | `/root/autodl-tmp/nerf_llff_data`（脚本内路径） |
| Mip-NeRF360 数据 | `/root/autodl-tmp/Mip-NeRF360/360_v2`（脚本内路径） |
| 训练输出 | 仅写 `output/`；日志写 `logs/`（不存在先 `mkdir -p logs`） |

⚠️ 首次在新实例工作先核实：`ls /root/autodl-tmp/`、`conda env list`、`nvidia-smi`。路径不符时改脚本头部的 `DATASET`/`REPO_DIR` 变量，**不要改 Python 代码**。
⚠️ `cloud_scripts/README.md` 里的"目录结构"图是旧版（写的是 `/opt/Sparse_View` + `dataset/`），以本文件和脚本实际路径为准（该矛盾已列入治理任务 T0）。

## 3. 一键脚本索引（`cloud_scripts/`）

| 脚本 | 用途 | 用法 |
|------|------|------|
| `setup_env.sh` | 环境搭建（约 10 分钟） | `bash cloud_scripts/setup_env.sh` |
| `run_dropgaussian.sh` | baseline 复现（LLFF + Mip360） | `bash cloud_scripts/run_dropgaussian.sh quick\|full` |
| `run_ablation_llff.sh` | v2 消融，8 场景 × 4 配置 = 32 runs | `bash cloud_scripts/run_ablation_llff.sh` |
| `run_falsification.sh` | **Step 1 密度方向证伪**，4 场景 × 2 配置 = 8 runs | `bash cloud_scripts/run_falsification.sh` |

统一运行方式：`bash cloud_scripts/<脚本>.sh 2>&1 | tee logs/<名>_$(date +%m%d_%H%M).log`

## 4. 关键 env var

| 变量 | 含义 | 默认 |
|------|------|------|
| `ALPHA_GRAD` | 位置梯度权重 α | 0.3 |
| `BETA_OPGRAD` | opacity 梯度权重 β | 0.5 |
| `GAMMA_DENSITY` | 几何密度权重 γ | 0.2 |
| `DENSITY_INVERT` | 密度项方向反转（1 = 密集→重要） | 0 |

## 5. AI 行为准则（硬约束）

1. **只跑脚本，不改代码/超参**——除非本次指令明确允许；脚本失败时原样报告最后 50 行报错并停下，不要自行 patch
2. 永不动 `submodules/`（diff-gaussian-rasterization、simple-knn 等 CUDA 扩展）
3. 一切新文件只写 `output/` 与 `logs/`
4. 训练前 `git log -1` 记录 commit hash；**不要在未 pull 的旧代码上跑实验**
5. 中断的 run：先确认 `output/` 下已有目录的训练进度，再决定续跑或清掉，不要混用目录
6. 跑完回传：`output/<exp>/**/metrics_*.txt`、完整日志、如有代码改动 → `git diff`；渲染图像只按需回传
7. commit 信息必须如实描述实现内容（历史教训：`9f57d83` 信息与实现不符）

## 6. 当前状态（2026-09-13）

- **待执行**：Step 1 密度方向证伪（`run_falsification.sh` 就绪，判定标准见脚本末尾注释）
- **之后**：视判定结果进入 SfM support 实现（约 20 行）或直接进入主实验
- **禁止**：在本仓库内混跑其他 baseline 仓库的代码（对比实验用各自仓库与环境）
