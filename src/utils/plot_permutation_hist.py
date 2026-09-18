# -*- coding: utf-8 -*-
"""plot_permutation_hist.py —— 坐标置换检验结果的直方图可视化

基于**已有的**逐次置换结果（`results/local_results/*/permutations/*/permutation_fpr.csv`，
每方法 20 次，列：perm, seed, n_genes, n_sig, fpr），不重跑任何方法。

产出两张图（保存到 results/summary/figures/）：
  1. permutation_fpr_hist_by_method.png  ：按方法分面（4 面板），叠加各数据集的 20 次 FPR 直方图
  2. permutation_fpr_hist_by_dataset.png ：按数据集分面（5 面板），叠加各方法的 20 次 FPR 直方图

口径说明：
  - 只使用 results/summary/fpr.csv 中记录的"完整"数据集×方法组合（5 数据集，排除不完整目录）。
  - FPR = 该次坐标置换下的检出数 / 候选基因数；名义 α = 0.05（图中红色虚线）。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"
FIGDIR = SUMMARY / "figures"

METHOD_COLORS = {
    "spark":  "#0052A2",   # 深蓝
    "nnsvg":  "#EB7400",   # 橙
    "spagcn": "#006B3F",   # 深绿
    "spaseg": "#805190",   # 紫
}
METHOD_LABELS = {
    "spark":  "SPARK-X",
    "nnsvg":  "nnSVG",
    "spagcn": "SpaGCN",
    "spaseg": "SpaSEG",
}
METHOD_ORDER = ["spark", "nnsvg", "spagcn", "spaseg"]
ALPHA = 0.05

# 图内短标签（避免图例过长；图中文字统一英文，与项目其他图一致）
DATASET_SHORT = {
    "DLPFC_151507": "DLPFC_151507",
    "mouse_brain_STARmap": "STARmap_brain",
    "STARmap_AD_8m_ctrl_rep1": "STARmap_AD_rep1",
    "Stereo_seq_drosophila": "Stereo-seq_fly",
    "Visium_Mouse_Olfactory_Bulb": "Visium_OB",
}


def _ds_label(d: str) -> str:
    return DATASET_SHORT.get(d, d)


def _locate_perm_csv(dataset: str, method: str) -> Path | None:
    """在 dataset 目录下递归定位某方法的 permutation_fpr.csv（兼容嵌套目录）。"""
    base = RESULTS / "local_results" / dataset
    if not base.exists():
        return None
    for p in base.rglob("permutation_fpr.csv"):
        if p.parent.name == method and p.parent.parent.name == "permutations":
            return p
    return None


def load_all() -> pd.DataFrame:
    """读取 fpr.csv 中记录的所有 (dataset, method)，并加载各自的逐次置换 FPR。"""
    rows = []
    with open(SUMMARY / "fpr.csv") as f:
        for r in csv.DictReader(f):
            ds, m = r["dataset"], r["method"]
            csv_path = _locate_perm_csv(ds, m)
            if csv_path is None:
                print(f"[warn] 缺少逐次置换文件: {ds}/{m}，跳过")
                continue
            df = pd.read_csv(csv_path)
            df["dataset"] = ds
            df["method"] = m
            df["mean_fpr"] = float(r["mean_fpr"])
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def plot_by_method(df: pd.DataFrame) -> None:
    """按方法分面：每个面板 = 一个方法，叠加 5 个数据集的 20 次 FPR 直方图。"""
    datasets = sorted(df["dataset"].unique())
    palette = plt.get_cmap("tab10")
    ds_colors = {d: palette(i % 10) for i, d in enumerate(datasets)}

    ncols = 4
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.6, 4.0), sharey=True)
    for ax, m in zip(axes, METHOD_ORDER):
        sub = df[df["method"] == m]
        if sub.empty:
            ax.set_title(METHOD_LABELS[m])
            ax.set_visible(False)
            continue
        xmax = 0.75
        for d in datasets:
            s = sub[sub["dataset"] == d]
            if s.empty:
                continue
            ax.hist(s["fpr"], bins=np.linspace(0, 1.0, 21), alpha=0.55,
                    color=ds_colors[d],
                    label=f"{_ds_label(d)} (mean {s['mean_fpr'].iloc[0]:.3f})")
        ax.axvline(ALPHA, ls="--", color="red", lw=1.2)
        ax.text(ALPHA, ax.get_ylim()[1] * 0.95, "alpha = 0.05", color="red",
                fontsize=8, ha="left", va="top")
        ax.set_xlim(0, xmax)
        ax.set_title(f"{METHOD_LABELS[m]}", fontsize=12)
        ax.set_xlabel("Per-run FPR")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("Count (20 runs/dataset)")
    fig.suptitle("Permutation-test FPR distribution (by method; 20 runs per dataset)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "permutation_fpr_hist_by_method.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成: {out}")


def plot_by_dataset(df: pd.DataFrame) -> None:
    """按数据集分面：每个面板 = 一个数据集，叠加各方法的 20 次 FPR 直方图。"""
    datasets = sorted(df["dataset"].unique())
    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.4, 4.0), sharey=True)
    for ax, d in zip(axes, datasets):
        sub = df[df["dataset"] == d]
        xmax = 0.75
        for m in METHOD_ORDER:
            s = sub[sub["method"] == m]
            if s.empty:
                continue
            ax.hist(s["fpr"], bins=np.linspace(0, 1.0, 21), alpha=0.5,
                    color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        ax.axvline(ALPHA, ls="--", color="red", lw=1.2)
        ax.text(ALPHA, ax.get_ylim()[1] * 0.95, "alpha = 0.05", color="red",
                fontsize=8, ha="left", va="top")
        ax.set_xlim(0, xmax)
        ax.set_title(d, fontsize=10)
        ax.set_xlabel("Per-run FPR")
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("Count (20 runs)")
    fig.suptitle("Permutation-test FPR distribution (by dataset; 20 runs per method)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "permutation_fpr_hist_by_dataset.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已生成: {out}")


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    if df.empty:
        print("[error] 未读到任何逐次置换结果")
        return 1
    print(f"读入 {df.shape[0]} 行（{df['dataset'].nunique()} 数据集 × {df['method'].nunique()} 方法 × 20 次）")
    plot_by_method(df)
    plot_by_dataset(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
