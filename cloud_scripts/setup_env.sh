#!/bin/bash
# ============================================================
# 云平台一键环境搭建脚本
# 适用: DropGaussian + DDGS 共享 conda 环境
# 用法: bash setup_env.sh
# 注意: 请在 /opt/Sparse_View/ 所在目录执行
# ============================================================
set -e

echo "===== Step 1: 创建 DDGS conda 环境 ====="
cd /opt/Sparse_View/DDGS

if conda env list | grep -q "DDGS"; then
    echo "[SKIP] DDGS 环境已存在"
else
    conda env create -f environment.yaml
    echo "[OK] DDGS 环境创建完成"
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DDGS

echo "===== Step 2: 编译 CUDA 扩展 ====="

cd /opt/Sparse_View/DDGS/submodules/diff-gaussian-rasterization
pip install -e . 2>&1 | tail -3
echo "[OK] diff-gaussian-rasterization"

cd /opt/Sparse_View/DDGS/submodules/simple-knn
pip install -e . 2>&1 | tail -3
echo "[OK] simple-knn"

echo "===== Step 3: 验证 ====="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB')
"

echo "===== 完成 ====="
