#!/bin/bash
# ============================================================
# DropGaussian 完整复现脚本 (LLFF + Mip-NeRF360)
#
# 注意: 当前代码已包含自适应 dropout (v2) 改动。
# 如需原始 DropGaussian baseline，请 checkout 原始 commit:
#   git stash && git checkout <original_commit_hash>
#
# 用法: bash run_dropgaussian.sh [quick|full]
#   quick: 单场景快速验证 (~10min, 10000 iter)
#   full:  全量复现 (默认, 10000 iter)
# 数据集路径按云实例实际情况修改 DATASET_LLFF / DATASET_MIP
# ============================================================
set -e

MODE=${1:-full}
DATASET_LLFF="/root/autodl-tmp/nerf_llff_data"
DATASET_MIP="/root/autodl-tmp/Mip-NeRF360/360_v2"
N_ITERATIONS=10000
cd /root/DropGaussian_release

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

if [ "$MODE" == "quick" ]; then
    echo "===== [QUICK] DropGaussian 快速验证: LLFF fern ====="
    python train.py \
        -s $DATASET_LLFF/fern/ \
        -m output/quick/fern \
        --eval -r 8 --n_views 3 --iterations 10000
    python render.py -m output/quick/fern --eval -r 8
    python metric.py --path output/quick --iteration 10000
    echo "查看结果: cat output/quick/results.json"
    exit 0
fi

# ============ 完整 LLFF ============
echo "===== [FULL] DropGaussian on LLFF (3-view, 8 scenes) ====="
LLFF_SCENES=("fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex")
EXP="llff_full"

for scene in "${LLFF_SCENES[@]}"; do
    echo "--- Training: $scene ---"
    python train.py \
        -s $DATASET_LLFF/$scene/ \
        -m output/$EXP/$scene \
        --eval -r 8 --n_views 3 --iterations $N_ITERATIONS

    echo "--- Rendering: $scene ---"
    python render.py -m output/$EXP/$scene --eval -r 8
done

python metric.py --path output/$EXP --iteration $N_ITERATIONS
echo "[DONE] LLFF 全部完成"

# ============ 完整 Mip-NeRF360 ============
echo "===== [FULL] DropGaussian on Mip-NeRF360 (12-view, 7 scenes) ====="
MIP_SCENES=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
EXP="mip_full"

for scene in "${MIP_SCENES[@]}"; do
    echo "--- Training: $scene ---"
    python train.py \
        -s $DATASET_MIP/$scene/ \
        -m output/$EXP/$scene \
        --eval -r 8 --n_views 12 --iterations $N_ITERATIONS

    echo "--- Rendering: $scene ---"
    python render.py -m output/$EXP/$scene --eval -r 8
done

python metric.py --path output/$EXP --iteration $N_ITERATIONS
echo "[DONE] Mip-NeRF360 全部完成"

echo ""
echo "===== DropGaussian 复现完成 ====="
echo "LLFF results:   output/llff_full/results.json"
echo "Mip360 results: output/mip_full/results.json"
