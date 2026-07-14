#!/usr/bin/env python3
"""
DropGaussian vs DDGS 结果对比脚本

用法:
  python compare_results.py \\
    --dropgaussian /path/to/DropGaussian_release/output \\
    --ddgs /path/to/DDGS/output \\
    --output comparison.md

从各方法 output 目录递归收集 results.json，按场景做对比。
"""

import json
import os
import argparse
from collections import defaultdict
from pathlib import Path


def find_results(base_path: str) -> dict:
    """递归查找所有 results.json，返回 {scene_name: metrics_dict}"""
    results = {}
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f == "results.json":
                path = Path(root)
                # 场景名取最上层子目录
                rel = path.relative_to(base_path)
                scene = rel.parts[0] if rel.parts else "unknown"

                with open(path / f) as fp:
                    data = json.load(fp)

                # DDGS 有多个 seed —— 取平均
                if scene not in results:
                    results[scene] = []
                results[scene].append(data)
    # 对每个场景的多次运行取平均
    averaged = {}
    for scene, runs in results.items():
        if len(runs) == 1:
            averaged[scene] = runs[0]
        else:
            avg = {}
            for key in ["PSNR", "SSIM", "LPIPS"]:
                vals = [r.get(key, 0) for r in runs if key in r]
                avg[key] = sum(vals) / len(vals) if vals else 0
            avg["num_runs"] = len(runs)
            averaged[scene] = avg
    return averaged


def main():
    parser = argparse.ArgumentParser(description="Compare DropGaussian vs DDGS results")
    parser.add_argument("--dropgaussian", required=True, help="DropGaussian output directory")
    parser.add_argument("--ddgs", required=True, help="DDGS output directory")
    parser.add_argument("--output", default="comparison.md", help="Output markdown file")
    args = parser.parse_args()

    dg = find_results(args.dropgaussian)
    ddgs = find_results(args.ddgs)

    all_scenes = sorted(set(list(dg.keys()) + list(ddgs.keys())))

    lines = []
    lines.append("# DropGaussian vs DDGS 复现对比")
    lines.append("")
    lines.append("## 逐场景指标")
    lines.append("")
    lines.append("| Scene | DG-PSNR | DDGS-PSNR | ΔPSNR | DG-SSIM | DDGS-SSIM | DG-LPIPS | DDGS-LPIPS |")
    lines.append("|-------|---------|-----------|-------|---------|-----------|----------|------------|")

    dg_psnr_sum = dg_ssim_sum = dg_lpips_sum = 0
    ddgs_psnr_sum = ddgs_ssim_sum = ddgs_lpips_sum = 0
    count = 0

    for scene in all_scenes:
        d = dg.get(scene, {})
        dd = ddgs.get(scene, {})

        dg_p = d.get("PSNR", 0)
        dg_s = d.get("SSIM", 0)
        dg_l = d.get("LPIPS", 0)
        dd_p = dd.get("PSNR", 0)
        dd_s = dd.get("SSIM", 0)
        dd_l = dd.get("LPIPS", 0)
        delta = dd_p - dg_p

        lines.append(
            f"| {scene} | {dg_p:.2f} | {dd_p:.2f} | {delta:+.2f} | "
            f"{dg_s:.4f} | {dd_s:.4f} | {dg_l:.4f} | {dd_l:.4f} |"
        )

        if dg_p > 0:
            dg_psnr_sum += dg_p
            dg_ssim_sum += dg_s
            dg_lpips_sum += dg_l
            ddgs_psnr_sum += dd_p
            ddgs_ssim_sum += dd_s
            ddgs_lpips_sum += dd_l
            count += 1

    if count > 0:
        lines.append("")
        lines.append("## 均值汇总")
        lines.append("")
        dg_avg_p = dg_psnr_sum / count
        dg_avg_s = dg_ssim_sum / count
        dg_avg_l = dg_lpips_sum / count
        dd_avg_p = ddgs_psnr_sum / count
        dd_avg_s = ddgs_ssim_sum / count
        dd_avg_l = ddgs_lpips_sum / count

        lines.append("| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |")
        lines.append("|------|-------|-------|--------|")
        lines.append(f"| DropGaussian | {dg_avg_p:.2f} | {dg_avg_s:.4f} | {dg_avg_l:.4f} |")
        lines.append(f"| DDGS | {dd_avg_p:.2f} | {dd_avg_s:.4f} | {dd_avg_l:.4f} |")
        lines.append(f"| Δ | {dd_avg_p-dg_avg_p:+.2f} | {dd_avg_s-dg_avg_s:+.4f} | {dd_avg_l-dg_avg_l:+.4f} |")

        lines.append("")
        lines.append("## Baseline 选择建议")
        lines.append("")
        lines.append("根据你的标准（可修改性 > 性能 > 依赖复杂度 > 训练时间）：")
        lines.append("")
        lines.append("- **DropGaussian** 优势：极简（1 个超参数）、无先验（归因清晰）、代码量最少")
        lines.append("- **DDGS** 优势：稳定性更强、自带稳定性评估工具、失效模式分析完善")
        lines.append("")
        if dd_avg_p - dg_avg_p < 1.0:
            lines.append(
                f"DDGS 仅领先 {dd_avg_p-dg_avg_p:.2f} dB PSNR，"
                f"如果这个差距是用 5 个额外超参数 + COLMAP 稠密深度换来的，"
                f"DropGaussian 作为 baseline 的'归因清晰度'优势可能更重要。"
            )
        else:
            lines.append(
                f"DDGS 领先 {dd_avg_p-dg_avg_p:.2f} dB PSNR，差距显著。"
                f"即使代码更复杂，性能优势可能让它成为更好的 baseline。"
            )

    with open(args.output, "w") as fp:
        fp.write("\n".join(lines))
    print(f"对比报告已保存到: {args.output}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
