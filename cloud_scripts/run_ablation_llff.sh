#!/bin/bash
# ============================================================
# 消融实验 v2：基于 Opacity 梯度的功能冗余度 Dropout
# LLFF 全场景 (8 scenes × 4 configs = 32 runs)
#
# v2 三信号公式:
#   importance = α × position_grad + β × opacity_grad + γ × (1-density)
#
# 实验设计:
#   baseline     = 所有信号权重为 0 → 均匀 dropout（≈原始盲 dropout）
#   opgrad_only  = 纯 opacity 梯度引导 (β=1.0)
#   opgrad_grad  = opacity梯度 + 位置梯度 (α=0.5, β=0.5)
#   all_three    = 三信号联合默认配置 (α=0.3, β=0.5, γ=0.2)
#
# 用法: bash run_ablation_llff.sh
# ============================================================
set -e

SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
EXPERIMENTS=("baseline" "opgrad_only" "opgrad_grad" "all_three")

# 实验配置: ALPHA_GRAD BETA_OPGRAD GAMMA_DENSITY
declare -A ALPHAS
declare -A BETAS
declare -A GAMMAS

ALPHAS[baseline]=0
BETAS[baseline]=0
GAMMAS[baseline]=0

ALPHAS[opgrad_only]=0
BETAS[opgrad_only]=1.0
GAMMAS[opgrad_only]=0

ALPHAS[opgrad_grad]=0.5
BETAS[opgrad_grad]=0.5
GAMMAS[opgrad_grad]=0

ALPHAS[all_three]=0.3
BETAS[all_three]=0.5
GAMMAS[all_three]=0.2

DATASET="/root/autodl-tmp/nerf_llff_data"
N_VIEWS=3
N_ITER=10000

cd /root/DropGaussian_release

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

total=$(( ${#SCENES[@]} * ${#EXPERIMENTS[@]} ))
count=0

for exp in "${EXPERIMENTS[@]}"; do
    alpha="${ALPHAS[$exp]}"
    beta="${BETAS[$exp]}"
    gamma="${GAMMAS[$exp]}"

    for scene in "${SCENES[@]}"; do
        count=$((count + 1))
        echo "===== [$count/$total] $exp | $scene (α=$alpha β=$beta γ=$gamma) ====="

        export ALPHA_GRAD=$alpha
        export BETA_OPGRAD=$beta
        export GAMMA_DENSITY=$gamma

        python train.py \
            -s $DATASET/$scene/ \
            -m output/ablation_v2/$exp/$scene \
            --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER

        python render.py -m output/ablation_v2/$exp/$scene --eval -r 8

        # Quick score
        metrics_file="output/ablation_v2/$exp/$scene/metrics_${N_ITER}.txt"
        if [ -f "$metrics_file" ]; then
            psnr=$(head -1 "$metrics_file" | sed 's/[^0-9.]//g')
            echo "  → PSNR: $psnr"
        fi
    done
done

# ============================================================
# 生成 metrics (metrics_mean.txt)
# ============================================================
echo ""
echo "===== 生成 metrics ====="
for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="output/ablation_v2/$exp"
    if [ -d "$exp_dir" ]; then
        echo "--- $exp ---"
        python metric.py --path "$exp_dir" --iteration $N_ITER
    fi
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "=============================================="
echo "  消融实验 v2 汇总 (LLFF 3-view, ${N_ITER} iter)"
echo "=============================================="
echo "  公式: importance = α·grad + β·opacity_grad + γ·(1-density)"
echo ""
printf "%-14s" "Scene"
for exp in "${EXPERIMENTS[@]}"; do
    printf "%-14s" "$exp"
done
echo ""

for scene in "${SCENES[@]}"; do
    printf "%-14s" "$scene"
    for exp in "${EXPERIMENTS[@]}"; do
        f="output/ablation_v2/$exp/$scene/metrics_${N_ITER}.txt"
        if [ -f "$f" ]; then
            psnr=$(head -1 "$f" | sed 's/[^0-9.]//g')
            printf "%-14s" "$psnr"
        else
            printf "%-14s" "N/A"
        fi
    done
    echo ""
done
echo "=============================================="
