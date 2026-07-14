#!/bin/bash
# ============================================================
# 消融实验：LLFF 全场景 (6 scenes × 4 configs = 24 runs)
#
# 实验设计:
#   baseline    = 原版 DropGaussian (盲 dropout)
#   grad_only   = 仅梯度引导 (ALPHA_GRAD=1.0)
#   density_only= 仅密度引导 (ALPHA_GRAD=0.0)
#   combined    = 梯度+密度联合 (ALPHA_GRAD=0.5)
#
# 用法: bash run_ablation_llff.sh
# ============================================================
set -e

# SCENES=("fern" "flower" "horns" "leaves" "orchids" "trex")
SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
EXPERIMENTS=("baseline" "grad_only" "density_only" "combined")
ALPHAS=(0.5 1.0 0.0 0.5)  # for baseline alpha doesn't matter
# DATASET="dataset/nerf_llff_data"
DATASET="/root/autodl-tmp/nerf_llff_data"
N_VIEWS=3
N_ITER=30000
cd /root/DropGaussian_release

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

total=$(( ${#SCENES[@]} * ${#EXPERIMENTS[@]} ))
count=0

for exp_idx in "${!EXPERIMENTS[@]}"; do
    exp="${EXPERIMENTS[$exp_idx]}"
    alpha="${ALPHAS[$exp_idx]}"

    for scene in "${SCENES[@]}"; do
        count=$((count + 1))
        echo "===== [$count/$total] $exp | $scene (alpha=$alpha) ====="

        export ALPHA_GRAD=$alpha

        python train.py \
            -s $DATASET/$scene/ \
            -m output/ablation/$exp/$scene \
            --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER

        python render.py -m output/ablation/$exp/$scene --eval -r 8

        # Quick score
        result_file="output/ablation/$exp/$scene/results.json"
        if [ -f "$result_file" ]; then
            psnr=$(python -c "import json; print(json.load(open('$result_file')).get('PSNR', 'N/A'))" 2>/dev/null)
            echo "  → PSNR: $psnr"
        fi
    done
done

# ============================================================
# 生成 metrics (results.json)
# ============================================================
echo ""
echo "===== 生成 metrics ====="
for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="output/ablation/$exp"
    if [ -d "$exp_dir" ]; then
        echo "--- $exp ---"
        python metric.py --path "$exp_dir" --iteration 30000
    fi
done

# ============================================================
# 汇总
# ============================================================
echo ""
echo "=============================================="
echo "  消融实验汇总 (LLFF 3-view)"
echo "=============================================="
printf "%-14s" "Scene"
for exp in "${EXPERIMENTS[@]}"; do
    printf "%-12s" "$exp"
done
echo ""

for scene in "${SCENES[@]}"; do
    printf "%-14s" "$scene"
    for exp in "${EXPERIMENTS[@]}"; do
        f="output/ablation/$exp/$scene/results.json"
        if [ -f "$f" ]; then
            psnr=$(python -c "import json; print(f'{json.load(open(\"$f\")).get(\"PSNR\",0):.2f}')" 2>/dev/null || echo "  N/A")
            printf "%-12s" "$psnr"
        else
            printf "%-12s" "N/A"
        fi
    done
    echo ""
done
echo "=============================================="
