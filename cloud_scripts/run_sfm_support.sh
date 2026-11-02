#!/bin/bash
# ============================================================
# T2：SfM support 第四信号消融
# LLFF 4 场景 × 3 配置 = 12 runs（先砍半试水，全量 8 场景视结果决定）
#
# 公式（DELTA_SFM > 0 时）:
#   importance = α·position_grad + β·opacity_grad + γ·density_term + δ·sfm_support
#
# 配置:
#   v1_base  = α=0.5, β=0, γ=0.5, δ=0    （v1 Combined 对照）
#   v1_sfm   = α=0.5, β=0, γ=0,   δ=0.5  （用 SfM support 替换密度项）
#   sfm_only = α=0,   β=0, γ=0,   δ=1.0  （仅 SfM support）
#
# 用法: bash run_sfm_support.sh
# 注意: 输出目录为 output/sfm_support/<config>/<scene>
# ============================================================
set -e

SCENES=("trex" "room" "flower" "horns")
EXPERIMENTS=("v1_base" "v1_sfm" "sfm_only")

# ALPHA_GRAD BETA_OPGRAD GAMMA_DENSITY DELTA_SFM
declare -A ALPHAS
declare -A BETAS
declare -A GAMMAS
declare -A DELTAS

ALPHAS[v1_base]=0.5;  BETAS[v1_base]=0;  GAMMAS[v1_base]=0.5;  DELTAS[v1_base]=0
ALPHAS[v1_sfm]=0.5;   BETAS[v1_sfm]=0;   GAMMAS[v1_sfm]=0;     DELTAS[v1_sfm]=0.5
ALPHAS[sfm_only]=0;   BETAS[sfm_only]=0; GAMMAS[sfm_only]=0;   DELTAS[sfm_only]=1.0

DATASET="/root/autodl-tmp/nerf_llff_data"
N_VIEWS=3
N_ITER=10000

cd /root/DropGaussian_release

# conda may not be on PATH in non-interactive SSH; fall back to the known prefix.
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    source /root/miniconda3/etc/profile.d/conda.sh
fi
conda activate DDGS

total=$(( ${#SCENES[@]} * ${#EXPERIMENTS[@]} ))
count=0

for exp in "${EXPERIMENTS[@]}"; do
    alpha="${ALPHAS[$exp]}"
    beta="${BETAS[$exp]}"
    gamma="${GAMMAS[$exp]}"
    delta="${DELTAS[$exp]}"

    for scene in "${SCENES[@]}"; do
        count=$((count + 1))
        echo "===== [$count/$total] $exp | $scene (α=$alpha β=$beta γ=$gamma δ=$delta) ====="

        export ALPHA_GRAD=$alpha
        export BETA_OPGRAD=$beta
        export GAMMA_DENSITY=$gamma
        export DELTA_SFM=$delta

        python train.py \
            -s $DATASET/$scene/ \
            -m output/sfm_support/$exp/$scene \
            --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER

        python render.py -m output/sfm_support/$exp/$scene --eval -r 8

        metrics_file="output/sfm_support/$exp/$scene/metrics_${N_ITER}.txt"
        if [ -f "$metrics_file" ]; then
            psnr=$(head -1 "$metrics_file" | sed 's/[^0-9.]//g')
            echo "  → PSNR: $psnr"
        fi
    done
done

# ============================================================
# 生成 metrics
# ============================================================
echo ""
echo "===== 生成 metrics ====="
for exp in "${EXPERIMENTS[@]}"; do
    exp_dir="output/sfm_support/$exp"
    if [ -d "$exp_dir" ]; then
        echo "--- $exp ---"
        python metric.py --path "$exp_dir" --iteration $N_ITER
    fi
done

# ============================================================
# 汇总（PSNR）
# ============================================================
echo ""
echo "=============================================="
echo "  SfM support 消融汇总 (LLFF 3-view, ${N_ITER} iter)"
echo "=============================================="
echo "  公式: importance = α·grad + β·opacity_grad + γ·density + δ·sfm_support"
echo ""
printf "%-12s" "Scene"
for exp in "${EXPERIMENTS[@]}"; do
    printf "%-14s" "$exp"
done
echo ""

for scene in "${SCENES[@]}"; do
    printf "%-12s" "$scene"
    for exp in "${EXPERIMENTS[@]}"; do
        f="output/sfm_support/$exp/$scene/metrics_${N_ITER}.txt"
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
echo "回传清单: output/sfm_support/**/metrics_${N_ITER}.txt + 完整日志 + git status"
