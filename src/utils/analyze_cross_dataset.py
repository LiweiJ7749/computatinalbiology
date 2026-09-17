# -*- coding: utf-8 -*-
"""analyze_cross_dataset.py —— 跨数据集补充分析（§5 的 1/2/3 项）

基于 ``results/summary/metrics_long.csv`` + ``datasets.csv`` 产出：
  1. 统计校准汇总  calibration_summary.csv + calibration_*.png（QQ λ / KS_p 箱线）
  2. 效率-规模图    efficiency_scaling.png + scaling_fit.csv（双对数 + 斜率）
  3. 稳健性量化    robustness.csv（STARmap AD 8 样本 / DLPFC 4 切片 的跨重复 CV）

只 import 标准库 + numpy/pandas/matplotlib，方法色/标签复用 src。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402

ALL_METHODS = list(src.ALL_METHODS)   # spark, nnsvg, spagcn, spaseg


def _setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11})


def _color(m): return src.METHOD_COLORS.get(m, "#8C8C8C")
def _label(m): return src.METHOD_LABELS.get(m, "Random")


# ---------------------------------------------------------------------------
# 1. 统计校准汇总
# ---------------------------------------------------------------------------
def calibration(long: pd.DataFrame, tables_dir: Path, figdir: Path):
    df = long[long["method"].isin(ALL_METHODS) & long["metric"].isin(
        ["qq_lambda", "ks_uniform_p"])].copy()
    piv = df.pivot_table(index=["dataset", "tech", "platform_group", "dim"],
                         columns=["method", "metric"], values="value")
    piv = piv.reset_index()
    piv.to_csv(tables_dir / "calibration_summary.csv", index=False)

    for metric, title, fname in (
        ("qq_lambda", "QQ inflation lambda (lower ~ better calibrated)", "calibration_lambda.png"),
        ("ks_uniform_p", "KS uniformity p (higher ~ closer to uniform)", "calibration_ks.png"),
    ):
        sub = long[(long["method"].isin(ALL_METHODS)) & (long["metric"] == metric)]
        if sub.empty:
            continue
        _setup_style()
        fig, ax = plt.subplots(figsize=(6, 4))
        data = [sub[sub["method"] == m]["value"].dropna().to_numpy(np.float64)
                for m in ALL_METHODS]
        positions = [1 + i for i in range(len(ALL_METHODS))]
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False)
        for patch, m in zip(bp["boxes"], ALL_METHODS):
            patch.set_facecolor(_color(m)); patch.set_alpha(0.7)
        for m, d in zip(ALL_METHODS, data):
            if len(d):
                ax.scatter(np.full(len(d), positions[ALL_METHODS.index(m)]),
                           d, s=12, color=_color(m), alpha=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels([_label(m) for m in ALL_METHODS])
        ax.set_ylabel(title)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figdir / fname, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 汇总（每方法的中位数）
    summ = sub.groupby("method")["value"].median() if not sub.empty else None
    src.log_message("校准汇总已产出: calibration_summary.csv + 2 图")
    return piv


# ---------------------------------------------------------------------------
# 2. 效率-规模图
# ---------------------------------------------------------------------------
def efficiency(long: pd.DataFrame, datasets: pd.DataFrame, tables_dir: Path,
               figdir: Path):
    wall = long[(long["method"].isin(ALL_METHODS)) & (long["metric"] == "wall_seconds")]\
        .dropna(subset=["value"])
    wall = wall.merge(datasets[["dataset", "n_spots", "n_genes"]], on="dataset")
    wall = wall[wall["value"] > 0]

    _setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.8))
    fits = []
    for m in ALL_METHODS:
        sub = wall[wall["method"] == m]
        if len(sub) < 2:
            continue
        x = np.log10(sub["n_spots"].to_numpy(np.float64))
        y = np.log10(sub["value"].to_numpy(np.float64))
        ax.scatter(x, y, s=30, color=_color(m), label=_label(m), zorder=3, alpha=0.8)
        b, a = np.polyfit(x, y, 1)          # y = b*x + a
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, b * xs + a, color=_color(m), lw=1, ls="--", alpha=0.7)
        fits.append({"method": m, "n_points": int(len(sub)),
                     "slope_log10": float(b), "intercept_log10": float(a)})
    ax.set_xlabel("log10(n_spots)")
    ax.set_ylabel("log10(wall_seconds)")
    ax.set_title("Efficiency scaling (wall-clock vs spots)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "efficiency_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(fits).to_csv(tables_dir / "scaling_fit.csv", index=False)
    src.log_message("效率-规模图已产出: efficiency_scaling.png + scaling_fit.csv")
    return wall


# ---------------------------------------------------------------------------
# 3. 稳健性量化（跨重复 CV）
# ---------------------------------------------------------------------------
def robustness(long: pd.DataFrame, tables_dir: Path):
    metrics = ["median_moran_I", "moran_effect_z", "rank_vs_moran_rho"]
    df = long[(long["method"].isin(ALL_METHODS)) & (long["metric"].isin(metrics))]\
        .dropna(subset=["value"]).copy()

    # 分组：STARmap AD 8 样本（生物学重复）与 DLPFC 4 切片（邻近切片）
    df["group"] = np.where(df["dataset"].str.startswith("STARmap_AD"),
                           "STARmap_AD_8samples",
                           np.where(df["dataset"].str.startswith("DLPFC"),
                                    "DLPFC_4slices", None))
    rows = []
    for (grp, method, metric), sub in df[df["group"].notna()].groupby(
            ["group", "method", "metric"]):
        v = sub["value"].to_numpy(np.float64)
        mean = float(np.mean(v))
        std = float(np.std(v))
        cv = abs(std / mean) if mean != 0 else np.nan
        rows.append({"group": grp, "method": method, "metric": metric,
                     "n": int(len(v)), "mean": mean, "std": std, "cv": cv})
    rob = pd.DataFrame(rows)
    rob = rob.sort_values(["group", "metric", "method"])
    rob.to_csv(tables_dir / "robustness.csv", index=False)

    # 打印摘要（STARmap AD 生物重复，最严格口径）
    src.log_header("稳健性（跨重复 CV，越小越稳健）")
    for grp in ["STARmap_AD_8samples", "DLPFC_4slices"]:
        sub = rob[rob["group"] == grp]
        if sub.empty:
            continue
        print(f"\n[{grp}]  median_moran_I 的 CV：")
        for m in ALL_METHODS:
            r = sub[(sub["method"] == m) & (sub["metric"] == "median_moran_I")]
            if len(r):
                row = r.iloc[0]
                print(f"  {_label(m):8s} n={int(row['n'])} "
                      f"mean={row['mean']:.4f} cv={row['cv']:.3f}")
    src.log_message("稳健性结果已产出: robustness.csv")
    return rob


# ---------------------------------------------------------------------------
# 4. 空间权重口径稳健性（多 k 邻域，聚合各数据集 eval/tables/moran_k_robustness.csv）
# ---------------------------------------------------------------------------
def k_robustness(pairs: list, tables_dir: Path, figdir: Path):
    """聚合多 k 邻域稳健性：方法排名是否随 k（4/6/8/10/12）翻转。

    pairs: [(dataset, Path)] —— 各数据集的 moran_k_robustness.csv 路径。
    返回长表 DataFrame（dataset × k × method × n_sig × median_moran × rank）。
    """
    rows = []
    for ds, path in pairs:
        d = pd.read_csv(path)
        for k, sub in d.groupby("k"):
            v = sub["median_moran"].to_numpy(np.float64)
            # 越大越好 -> 降序排名（nan 不参与）
            order = v.argsort()[::-1]
            ranks = np.empty(len(v), dtype=np.float64)
            ranks[order] = np.arange(1, len(v) + 1)
            for (_, r), rk in zip(sub.iterrows(), ranks):
                rows.append({"dataset": ds, "k": int(k), "method": r["method"],
                             "n_sig": r["n_sig"], "median_moran": r["median_moran"],
                             "rank_within_dataset": rk})
    if not rows:
        return pd.DataFrame()
    kr = pd.DataFrame(rows).sort_values(["dataset", "k", "method"])
    kr.to_csv(tables_dir / "k_robustness.csv", index=False)

    _setup_style()
    datasets = list(dict.fromkeys(kr["dataset"].tolist()))
    ks = sorted(kr["k"].unique())
    methods = [m for m in ALL_METHODS if m in set(kr["method"])]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for ds in datasets:
        sub = kr[kr["dataset"] == ds]
        for m in methods:
            s = sub[sub["method"] == m]
            if s.empty:
                continue
            ax.plot(s["k"], s["median_moran"], marker="o", ms=3,
                    color=_color(m), alpha=0.45, lw=0.8)
    # 叠加每个方法跨数据集的均值曲线（加粗）
    for m in methods:
        s = kr[kr["method"] == m].groupby("k")["median_moran"].mean()
        ax.plot(s.index, s.values, marker="s", ms=5, lw=2.2,
                color=_color(m), label=_label(m), zorder=5)
    ax.set_xlabel("k (spatial neighbors)")
    ax.set_ylabel("Median Moran's I of significant SVGs")
    ax.set_title("Spatial-weight robustness (thin = per dataset, thick = mean)")
    ax.set_xticks(ks)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "k_robustness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 打印：排名随 k 是否变化（翻转次数）
    src.log_header("空间权重口径稳健性（k=4..12）")
    for ds in datasets:
        sub = kr[kr["dataset"] == ds]
        flips = 0
        prev = None
        for k in ks:
            order = tuple(sub[sub["k"] == k]
                          .sort_values("rank_within_dataset")["method"].tolist())
            if prev is not None and order != prev:
                flips += 1
            prev = order
        cv = (sub.groupby("method")["median_moran"]
              .agg(lambda x: abs(x.std() / x.mean()) if x.mean() else np.nan))
        cv_s = ", ".join(f"{m}={cv[m]:.3f}" for m in methods if m in cv.index)
        print(f"[{ds}] 排名翻转次数={flips}/{len(ks)-1}  CV: {cv_s}")
    src.log_message(f"k 稳健性已产出: {tables_dir / 'k_robustness.csv'} + "
                    f"{figdir / 'k_robustness.png'}")
    return kr


def main() -> int:
    ap = argparse.ArgumentParser(description="跨数据集补充分析（校准/效率/稳健性）")
    ap.add_argument("--summary-dir", default=str(ROOT / "results" / "summary"))
    args = ap.parse_args()

    summary = Path(args.summary_dir)
    tables_dir = summary
    figdir = summary / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    long = pd.read_csv(summary / "metrics_long.csv")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")   # 混合列统一成数值
    datasets = pd.read_csv(summary / "datasets.csv")

    calibration(long, tables_dir, figdir)
    efficiency(long, datasets, tables_dir, figdir)
    robustness(long, tables_dir)

    # 多 k 邻域稳健性：扫描各数据集 eval/tables/moran_k_robustness.csv（存在才聚合）
    results_dir = ROOT / "results" / "local_results"
    pairs = []
    for sj in sorted(results_dir.rglob("eval/summary.json")):
        kpath = sj.parent / "tables" / "moran_k_robustness.csv"
        if kpath.exists():
            import json
            ds_key = json.loads(sj.read_text(encoding="utf-8")).get("dataset") \
                or sj.parent.parent.name
            pairs.append((ds_key, kpath))
    if pairs:
        src.log_message(f"发现 {len(pairs)} 个数据集有多 k 稳健性数据")
        k_robustness(pairs, tables_dir, figdir)
    else:
        src.log_message("未发现 moran_k_robustness.csv，跳过 k 稳健性")
    return 0


if __name__ == "__main__":
    sys.exit(main())