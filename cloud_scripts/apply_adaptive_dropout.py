#!/usr/bin/env python3
"""
应用自适应重要性 Dropout 修改到 DropGaussian 代码

用法（在云实例上）:
  python apply_adaptive_dropout.py /opt/Sparse_View/DropGaussian_release

这会:
  1. 在 scene/gaussian_model.py 添加 compute_importance_scores() 方法
  2. 在 gaussian_renderer/__init__.py 替换盲 dropout 为自适应 dropout
  3. 自动备份原文件为 .bak
"""

import sys
import os
import shutil


def patch_gaussian_model(path: str):
    """在 GaussianModel 类中添加 compute_importance_scores 方法"""
    filepath = os.path.join(path, "scene", "gaussian_model.py")
    shutil.copy2(filepath, filepath + ".bak")
    print(f"[OK] 备份: {filepath}.bak")

    with open(filepath, "r") as f:
        content = f.read()

    # 在 add_densification_stats 方法后面插入新方法
    old_method = '''    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1'''

    new_method = '''    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def compute_importance_scores(self):
        """
        Compute per-Gaussian importance scores for adaptive dropout.

        Two signals, no depth prior needed:
        1. Gradient activity: large view-space gradient → actively learning → important
        2. Local density:    small nearest-neighbor distance → crowded → redundant

        Returns:
            importance: Tensor of shape (N,) in [0, 1], higher = more important
        """
        N = self.get_xyz.shape[0]

        # ---- Signal 1: gradient activity ----
        # xyz_gradient_accum / denom = mean gradient norm per Gaussian
        if self.denom.sum() > 0:
            grad_active = self.xyz_gradient_accum / (self.denom + 1e-8)
            grad_active = grad_active.squeeze()
        else:
            grad_active = torch.zeros(N, device="cuda")

        g_max = grad_active.max()
        if g_max > 1e-8:
            grad_active = grad_active / g_max
        else:
            grad_active = torch.zeros(N, device="cuda")

        # ---- Signal 2: local density ----
        # distCUDA2 returns squared distance to nearest neighbor (already imported)
        nn_dist = distCUDA2(self.get_xyz)
        nn_dist = torch.clamp_min(nn_dist, 1e-8)
        density = 1.0 / nn_dist  # small distance → high density

        d_max = density.max()
        if d_max > 1e-8:
            density = density / d_max
        else:
            density = torch.zeros(N, device="cuda")

        # ---- Combine ----
        # High grad → important.  High density → redundant.
        # Read alpha from env var ALPHA_GRAD (default 0.5 for combined)
        alpha = float(os.environ.get("ALPHA_GRAD", 0.5))
        importance = alpha * grad_active + (1.0 - alpha) * (1.0 - density)
        importance = torch.clamp(importance, 0.0, 1.0)

        return importance'''

    if old_method not in content:
        print("[ERROR] 未找到 add_densification_stats 方法——文件可能已被修改")
        sys.exit(1)

    content = content.replace(old_method, new_method)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"[OK] 已添加 compute_importance_scores() → {filepath}")


def patch_renderer(path: str):
    """替换盲 dropout 为自适应 dropout"""
    filepath = os.path.join(path, "gaussian_renderer", "__init__.py")
    shutil.copy2(filepath, filepath + ".bak")
    print(f"[OK] 备份: {filepath}.bak")

    with open(filepath, "r") as f:
        content = f.read()

    old_block = '''    # DropGaussian
    if is_train:
        # Create initial compensation factor (1 for each Gaussian)
        compensation = torch.ones(opacity.shape[0], dtype=torch.float32, device="cuda")

        # Apply DropGaussian with compensation
        drop_rate = 0.2 * (iteration/10000)
        d = torch.nn.Dropout(p=drop_rate)
        compensation = d(compensation)

        # Apply to opacity
        opacity = opacity * compensation[:, None]'''

    new_block = '''    # ============================================================
    # Adaptive Importance Dropout (replaces blind DropGaussian dropout)
    # ============================================================
    if is_train:
        # Base dropout rate (same schedule as original DropGaussian)
        base_drop_rate = 0.2 * (iteration / 10000)

        # Get per-Gaussian importance scores
        importance = pc.compute_importance_scores()  # shape: (N,), in [0, 1]

        # Important Gaussians → lower dropout rate; redundant Gaussians → higher
        # importance → 1: drop_rate → 0 (fully protected)
        # importance → 0: drop_rate → base_drop_rate (full dropout)
        per_gaussian_drop_rate = base_drop_rate * (1.0 - importance)

        # Bernoulli sampling + compensation (same principle as original)
        keep_prob = 1.0 - per_gaussian_drop_rate
        keep_prob = torch.clamp(keep_prob, min=1e-8)  # avoid division by zero
        mask = torch.bernoulli(keep_prob)              # 1=keep, 0=drop
        compensation = mask / keep_prob                # survivors get boosted

        opacity = opacity * compensation[:, None]'''

    if old_block not in content:
        print("[ERROR] 未找到 DropGaussian dropout 代码块——文件可能已被修改")
        sys.exit(1)

    content = content.replace(old_block, new_block)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"[OK] 已替换为自适应 dropout → {filepath}")


def main():
    if len(sys.argv) < 2:
        print("用法: python apply_adaptive_dropout.py <DropGaussian_release 目录>")
        sys.exit(1)

    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"[ERROR] 目录不存在: {root}")
        sys.exit(1)

    print(f"===== 应用自适应重要性 Dropout =====")
    print(f"目标目录: {root}\n")

    patch_gaussian_model(root)
    patch_renderer(root)

    print(f"\n===== 完成 =====")
    print(f"原文件已备份为 .bak，如需恢复:")
    print(f"  mv {root}/scene/gaussian_model.py.bak {root}/scene/gaussian_model.py")
    print(f"  mv {root}/gaussian_renderer/__init__.py.bak {root}/gaussian_renderer/__init__.py")


if __name__ == "__main__":
    main()
