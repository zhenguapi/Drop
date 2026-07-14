#!/bin/bash
# 为已完成的消融实验生成 results.json
# 在云实例上执行: bash fix_generate_metrics.sh
set -e

cd /root/DropGaussian_release
source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

echo "===== 生成消融实验 metrics ====="
for exp in baseline grad_only density_only combined; do
    exp_dir="output/ablation/$exp"
    if [ -d "$exp_dir" ]; then
        echo "--- $exp ---"
        python metric.py --path "$exp_dir" --iteration 30000
    else
        echo "[SKIP] $exp_dir 不存在"
    fi
done

echo ""
echo "===== 汇总结果 ====="
echo ""
printf "%-14s" "Scene"
for exp in baseline grad_only density_only combined; do printf "%-12s" "$exp"; done
echo ""

# for scene in fern flower horns leaves orchids trex; do
for scene in fern flower fortress horns leaves orchids room trex; do
    printf "%-14s" "$scene"
    for exp in baseline grad_only density_only combined; do
        f="output/ablation/$exp/$scene/results.json"
        if [ -f "$f" ]; then
            psnr=$(python -c "import json; print(f'{json.load(open(\"$f\"))[\"PSNR\"]:.2f}')" 2>/dev/null || echo "N/A")
            printf "%-12s" "$psnr"
        else
            printf "%-12s" "N/A"
        fi
    done
    echo ""
done
echo "===== 完成 ====="
