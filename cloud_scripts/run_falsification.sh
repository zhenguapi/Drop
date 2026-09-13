#!/bin/bash
# ============================================================
# Step 1：密度方向证伪实验（DENSITY_INVERT）
#
# 目的：检验 v1 中密度信号（1-density）"方向反了"的假设。
#   配置 A density_normal : 密集 → 冗余（默认方向，即 v1 Combined 口径）
#   配置 B density_invert : 密集 → 重要（方向反转）
# 场景：trex/room（良观测，预期反转后下降）
#       flower/horns（差观测，预期反转后上升 ≥0.2 dB）
#
# 用法: bash run_falsification.sh
# 判定标准见脚本末尾汇总说明；完整任务卡在本地
# docs/tasks/T1_density_falsification/context.md（云端不可见）
#
# 数据集/代码路径按实例实际情况修改 DATASET / REPO_DIR
# ============================================================
set -e

REPO_DIR="/root/DropGaussian_release"
DATASET="/root/autodl-tmp/nerf_llff_data"
N_VIEWS=3
N_ITER=10000

SCENES=("trex" "room" "flower" "horns")
CONFIGS=("density_normal" "density_invert")

# 两配置共享 v1 Combined 权重，差异仅在 DENSITY_INVERT
ALPHA=0.5
BETA=0
GAMMA=0.5

cd $REPO_DIR

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

total=$(( ${#SCENES[@]} * ${#CONFIGS[@]} ))
count=0

for config in "${CONFIGS[@]}"; do
    if [ "$config" == "density_invert" ]; then invert=1; else invert=0; fi
    exp_dir="output/falsification/$config"

    for scene in "${SCENES[@]}"; do
        count=$((count + 1))
        echo "===== [$count/$total] $config | $scene (DENSITY_INVERT=$invert) ====="

        export ALPHA_GRAD=$ALPHA
        export BETA_OPGRAD=$BETA
        export GAMMA_DENSITY=$GAMMA
        export DENSITY_INVERT=$invert

        python train.py \
            -s $DATASET/$scene/ \
            -m $exp_dir/$scene \
            --eval -r 8 --n_views $N_VIEWS --iterations $N_ITER

        python render.py -m $exp_dir/$scene --eval -r 8

        # Quick score
        metrics_file="$exp_dir/$scene/metrics_${N_ITER}.txt"
        if [ -f "$metrics_file" ]; then
            psnr=$(head -1 "$metrics_file" | sed 's/[^0-9.]//g')
            echo "  → PSNR: $psnr"
        fi
    done

    python metric.py --path "$exp_dir" --iteration $N_ITER
done

# ============================================================
# 汇总：normal vs invert 逐场景对比
# ============================================================
echo ""
echo "=============================================="
echo "  密度方向证伪汇总 (LLFF 3-view, ${N_ITER} iter)"
echo "  公式: importance = ${ALPHA}·grad + ${GAMMA}·密度项(方向由 DENSITY_INVERT 决定)"
echo "=============================================="
printf "%-10s %-14s %-14s %-12s\n" "Scene" "normal" "invert" "D(invert)"

for scene in "${SCENES[@]}"; do
    fn="output/falsification/density_normal/$scene/metrics_${N_ITER}.txt"
    fi_="output/falsification/density_invert/$scene/metrics_${N_ITER}.txt"
    if [ -f "$fn" ] && [ -f "$fi_" ]; then
        psnr_n=$(head -1 "$fn" | sed 's/[^0-9.]//g')
        psnr_i=$(head -1 "$fi_" | sed 's/[^0-9.]//g')
        delta=$(awk -v a="$psnr_i" -v b="$psnr_n" 'BEGIN{printf "%.3f", a-b}')
        printf "%-10s %-14s %-14s %-12s\n" "$scene" "$psnr_n" "$psnr_i" "$delta"
    else
        printf "%-10s %-14s %-14s %-12s\n" "$scene" "N/A" "N/A" "N/A"
    fi
done
echo "=============================================="
echo "判定标准（阈值以 T0 决议为准，默认 ±0.1 dB）："
echo "  flower/horns D >= +0.2        → H1 支持，进入 Step 2 (SfM support)"
echo "  全部场景 |D| <= 0.1           → H0 成立，跳过 Step 2 直接主实验"
echo "  trex/room 大降且 flower/horns 不变 → 中间态，仍进 Step 2"
echo "回传清单: output/falsification/**/metrics_${N_ITER}.txt + 完整日志 + git status"
