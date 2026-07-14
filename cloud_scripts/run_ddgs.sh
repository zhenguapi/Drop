#!/bin/bash
# ============================================================
# DDGS (D²GS) 完整复现脚本 (LLFF + Mip-NeRF360)
# 用法: bash run_ddgs.sh [quick|full|single_run]
#   quick:      单场景快速验证
#   single_run: 全场景但每场景只跑 1 个 seed（快）
#   full:       全场景 × 3 seeds（论文标准，耗时 3 倍）
# ============================================================
set -e

MODE=${1:-single_run}
DATASET_LLFF="dataset/nerf_llff_data"
DATASET_MIP="dataset/mipnerf360"
N_ITERATIONS=30000
cd /opt/Sparse_View/DDGS

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

# ============================================================
# LLFF 每场景超参数（来自论文 scripts/train_llff.sh）
# 格式: depth_w:density_w:drop_min:drop_max:mask_param_lambda_far
# ============================================================
declare -A LLFF_CONFIGS=(
    ["fern"]="0.7:0.3:0.1:0.3:10_1.0"
    ["flower"]="0.7:0.3:0.1:0.3:5_1.5"
    ["fortress"]="0.5:0.5:0.05:0.3:15_1.0"
    ["horns"]="0.7:0.3:0.1:0.3:5_1.0"
    ["leaves"]="0.7:0.3:0.1:0.5:10_0.25"
    ["orchids"]="0.5:0.5:0.05:0.3:10_0.5"
    ["room"]="0.15:0.85:0.05:0.5:20_0.5"
    ["trex"]="0.15:0.85:0.05:0.5:5_0.5"
)

declare -A MIP_CONFIGS=(
    ["bicycle"]="0.15:0.85:0.05:0.5:20_1.0"
    ["bonsai"]="0.7:0.3:0.1:0.3:5_0.5"
    ["counter"]="0.15:0.85:0.05:0.5:5_0.5"
    ["garden"]="0.3:0.7:0.05:0.7:15_0.5"
    ["kitchen"]="0.1:0.9:0.05:0.5:10_0.5"
    ["room"]="0.3:0.7:0.05:0.7:5_0.5"
    ["stump"]="0.3:0.7:0.05:0.7:10_1.0"
)

run_training() {
    local dataset=$1 dataset_path=$2 n_views=$3 output_base=$4
    local config_value=$5
    local seed=${6:-1}

    IFS=':' read -r depth_w density_w drop_min drop_max mask_pair <<< "$config_value"
    IFS='_' read -r mask_param lambda_far <<< "$mask_pair"
    local drop_id="dw${depth_w}_dnw${density_w}_dmin${drop_min}_dmax${drop_max}"
    local run_output="${output_base}/${dataset}/${drop_id}/mask${mask_param}_lambda${lambda_far}/run_${seed}"

    mkdir -p "$run_output"

    echo "  [seed=$seed] depth_w=$depth_w density_w=$density_w drop_min=$drop_min drop_max=$drop_max mask=$mask_param lambda=$lambda_far"
    python train.py \
        -s "${dataset_path}/${dataset}" \
        -m "$run_output" \
        --depth_weight "$depth_w" \
        --density_weight "$density_w" \
        --drop_min "$drop_min" \
        --drop_max "$drop_max" \
        --mask_param "$mask_param" \
        --lambda_far "$lambda_far" \
        --eval -r 8 --n_views "$n_views" --iterations $N_ITERATIONS

    python render.py -m "$run_output" --eval -r 8
}

# ============================================================
# QUICK MODE
# ============================================================
if [ "$MODE" == "quick" ]; then
    echo "===== [QUICK] DDGS 快速验证: LLFF fern ====="
    run_training "fern" "$DATASET_LLFF" 3 "output/quick" "${LLFF_CONFIGS[fern]}" 1
    echo "查看结果: $(ls output/quick/fern/*/mask*/run_1/results.json 2>/dev/null)"
    exit 0
fi

# ============================================================
# LLFF 全量
# ============================================================
NUM_RUNS=1
if [ "$MODE" == "full" ]; then NUM_RUNS=3; fi

echo "===== DDGS on LLFF (3-view, 8 scenes, ${NUM_RUNS} seed(s)) ====="
for scene in "${!LLFF_CONFIGS[@]}"; do
    echo "--- Scene: $scene ---"
    for run in $(seq 1 $NUM_RUNS); do
        run_training "$scene" "$DATASET_LLFF" 3 "output/llff_full" "${LLFF_CONFIGS[$scene]}" "$run"
    done
done
echo "[DONE] LLFF"

# ============================================================
# Mip-NeRF360 全量
# ============================================================
echo "===== DDGS on Mip-NeRF360 (12-view, 7 scenes, ${NUM_RUNS} seed(s)) ====="
for scene in "${!MIP_CONFIGS[@]}"; do
    echo "--- Scene: $scene ---"
    for run in $(seq 1 $NUM_RUNS); do
        run_training "$scene" "$DATASET_MIP" 12 "output/mip_full" "${MIP_CONFIGS[$scene]}" "$run"
    done
done
echo "[DONE] Mip-NeRF360"

echo ""
echo "===== DDGS 复现完成 ====="
