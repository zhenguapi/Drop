#!/bin/bash
# ============================================================
# 消融实验：自适应重要性 Dropout
#
# 实验矩阵:
#   A: alpha=1.0  → 仅梯度引导 (grad-only)
#   B: alpha=0.0  → 仅密度引导 (density-only)
#   C: alpha=0.5  → 梯度+密度联合 (combined)
#   Baseline:     原版 DropGaussian (盲 dropout)
#
# 用法: bash run_ablation.sh [scene_name]
#   scene_name: fern, flower, horns, leaves, orchids, trex (默认 fern)
# ============================================================
set -e

SCENE=${1:-fern}
DATASET="/root/autodl-tmp/nerf_llff_data"
N_VIEWS=3
N_ITER=30000
cd /root/DropGaussian_release

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

echo "===== 消融实验: scene=$SCENE ====="

# --- Baseline: 原版盲 dropout ---
echo "[1/4] Baseline (盲 dropout)"
python train.py \
    -s $DATASET/$SCENE/ \
    -m output/ablation/baseline/$SCENE \
    --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER
python render.py -m output/ablation/baseline/$SCENE --eval -r 8

# --- 修改 alpha → 梯度 与 密度需要注入 ---
# 方案：用环境变量传递给 gaussian_model.py
# 在 compute_importance_scores() 中读取 ALPHA_GRAD

# --- 实验 A: 仅梯度引导 (alpha=1.0) ---
echo "[2/4] 仅梯度引导"
ALPHA_GRAD=1.0 python train.py \
    -s $DATASET/$SCENE/ \
    -m output/ablation/grad_only/$SCENE \
    --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER
python render.py -m output/ablation/grad_only/$SCENE --eval -r 8

# --- 实验 B: 仅密度引导 (alpha=0.0) ---
echo "[3/4] 仅密度引导"
ALPHA_GRAD=0.0 python train.py \
    -s $DATASET/$SCENE/ \
    -m output/ablation/density_only/$SCENE \
    --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER
python render.py -m output/ablation/density_only/$SCENE --eval -r 8

# --- 实验 C: 联合引导 (alpha=0.5) ---
echo "[4/4] 梯度+密度联合"
ALPHA_GRAD=0.5 python train.py \
    -s $DATASET/$SCENE/ \
    -m output/ablation/combined/$SCENE \
    --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER
python render.py -m output/ablation/combined/$SCENE --eval -r 8

# --- 收集结果 ---
echo ""
echo "===== 消融实验结果 ====="
echo "Scene: $SCENE"
for exp in baseline grad_only density_only combined; do
    result_file="output/ablation/$exp/$SCENE/results.json"
    if [ -f "$result_file" ]; then
        psnr=$(python -c "import json; d=json.load(open('$result_file')); print(d.get('PSNR','N/A'))" 2>/dev/null || echo "N/A")
        echo "  $exp: PSNR=$psnr"
    else
        echo "  $exp: (未生成)"
    fi
done
