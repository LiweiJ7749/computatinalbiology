# -*- coding: utf-8 -*-
"""plot_cross_dataset.py —— 跨数据集结果对比图（第②步）

基于 ``results/summary/metrics_wide.csv`` + ``datasets.csv`` 产出三类图：

  1. rank_heatmap.png   每指标的方法平均排名热图（方向统一为"排名 1 = 最好"）
  2. win_loss.png       方法两两 win-loss 矩阵（净胜场 / 可比单元数）
  3. platform_boxplot   平台分层箱线图（正确性指标 × platform_group，方法分色）

关键口径：
  * 排名只在"可比子集"内做：某方法在某数据集上该指标为 NaN 时，该单元不参与
    该指标的排名（避免把缺位当最差），其余方法照常排名。
  * 指标方向 ``METRIC_DIRECTION``：+1 = 越大越好，-1 = 越小越好。
  * 下游指标（ari/nmi/region_eta2）只在 ``has_labels=True`` 的数据集上有值，
    缺失即标记 NaN，不参与排名。
  * 分组箱线图 x = platform_group，方法用 src.METHOD_COLORS 分色，符合平台分层叙事。
  * 所有标签用英文（避免终端/图片 GBK 问题）；方法显示名用 src.METHOD_LABELS。
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

import src  # noqa: E402  (取 ALL_METHODS / METHOD_COLORS / METHOD_LABELS)

ALL_METHODS = list(src.ALL_METHODS)          # spark, nnsvg, spagcn, spaseg
RANDOM_KEY = "random"

# 指标方向：+1 越大越好；-1 越小越好。未列出的指标不参与排名/win-loss。
METRIC_DIRECTION = {
    "median_moran_I": 1,
    "median_geary_C_star": 1,
    "moran_effect_z": 1,
    "rank_vs_moran_rho": 1,
    "null_p": -1,
    "frac_low_spots": -1,
    "region_eta2": 1,
    "region_eta2_mean": 1,
    "ari_top100": 1,
    "nmi_top100": 1,
}

# 排名热图里展示的指标（顺序即热图行序）。
RANK_HEATMAP_METRICS = [
    "median_moran_I", "median_geary_C_star", "moran_effect_z",
    "rank_vs_moran_rho", "null_p", "frac_low_spots",
    "region_eta2", "region_eta2_mean", "ari_top100", "nmi_top100",
]

# 平台分层箱线图要画的正确性指标（每个指标一张图）。
BOXPLOT_METRICS = [
    "median_moran_I", "moran_effect_z", "median_geary_C_star", "rank_vs_moran_rho",
]

METRIC_LABEL = {
    "median_moran_I": "Median Moran's I",
    "median_geary_C_star": "Median Geary C*",
    "moran_effect_z": "Moran effect Z",
    "rank_vs_moran_rho": "Spearman rho (rank~Moran)",
    "null_p": "null p-value",
    "frac_low_spots": "frac. low-expr spots",
    "region_eta2": "Region eta^2 (median)",
    "region_eta2_mean": "Region eta^2 (mean)",
    "ari_top100": "ARI @ top-100",
    "nmi_top100": "NMI @ top-100",
}

# 平台分层显示顺序（保证出图稳定）。
PLATFORM_ORDER = ["array", "imaging", "high_density"]
PLATFORM_LABEL = {"array": "array", "imaging": "imaging", "high_density": "high-density"}


def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
    })


def load(wide_path: Path, datasets_path: Path):
    wide = pd.read_csv(wide_path)
    ds = pd.read_csv(datasets_path)
    # 合并 platform_group/tech（以 dataset 为键）
    meta = ds[["dataset", "tech", "platform_group", "dim", "has_labels"]]
    wide = wide.merge(meta, on="dataset", how="left")
    return wide


def _method_color(m: str) -> str:
    return src.METHOD_COLORS.get(m, "#8C8C8C")


def _method_label(m: str) -> str:
    return src.METHOD_LABELS.get(m, "Random")


# ---------------------------------------------------------------------------
# 排名与 win-loss 计算
# ---------------------------------------------------------------------------
def compute_rank_matrix(wide: pd.DataFrame, methods):
    """对每个 (metric, dataset) 单元内的方法排名（1=最好，NaN 单元跳过）。

    返回 (rank_mean: DataFrame[metric x method] 平均排名,
          rank_count: DataFrame[metric x method] 参与单元数)。
    """
    metrics = [m for m in RANK_HEATMAP_METRICS if m in METRIC_DIRECTION]
    rows = []
    for _, r in wide.iterrows():
        ds = r["dataset"]
        for metric in metrics:
            direction = METRIC_DIRECTION[metric]
            # 宽表列名：method__metric；随机基线不参与排名
            pairs = []
            for m in methods:
                c = f"{m}__{metric}"
                if c in wide.columns and pd.notna(r[c]):
                    pairs.append((m, float(r[c])))
            if len(pairs) < 2:
                continue
            vals = np.array([v for _, v in pairs], dtype=np.float64)
            # 越大越好 -> 降序排名（最小值/最大值得 rank1）；越小越好 -> 升序
            order = vals.argsort()
            if direction > 0:
                order = order[::-1]
            ranks = np.empty(len(vals), dtype=np.float64)
            ranks[order] = np.arange(1, len(vals) + 1)
            for (m, _), rk in zip(pairs, ranks):
                rows.append({"dataset": ds, "metric": metric, "method": m, "rank": rk})

    rdf = pd.DataFrame(rows)
    rank_mean = rdf.pivot_table(index="metric", columns="method", values="rank",
                                aggfunc="mean")
    rank_count = rdf.pivot_table(index="metric", columns="method", values="rank",
                                 aggfunc="count")
    # 固定顺序
    rank_mean = rank_mean.reindex(index=metrics, columns=methods)
    rank_count = rank_count.reindex(index=metrics, columns=methods)
    return rank_mean, rank_count, rdf


def compute_win_loss(wide: pd.DataFrame, methods):
    """在可比 (dataset, metric) 单元上统计方法两两胜负。

    返回 (net: DataFrame[method x method] net = wins - losses,
          total: DataFrame[method x method] 可比单元数 上三角)。
    """
    n = len(methods)
    wins = np.zeros((n, n), dtype=np.int32)
    total = np.zeros((n, n), dtype=np.int32)

    for _, r in wide.iterrows():
        for mi, a in enumerate(methods):
            for mj, b in enumerate(methods):
                if mi >= mj:
                    continue
                # 对每个有方向的指标比较 a vs b
                for metric, direction in METRIC_DIRECTION.items():
                    ca = f"{a}__{metric}"
                    cb = f"{b}__{metric}"
                    if ca not in wide.columns or cb not in wide.columns:
                        continue
                    va, vb = r[ca], r[cb]
                    if pd.isna(va) or pd.isna(vb):
                        continue
                    total[mi, mj] += 1
                    if direction > 0:
                        a_better = va > vb
                    else:
                        a_better = va < vb
                    if a_better:
                        wins[mi, mj] += 1
                    elif va == vb:
                        total[mi, mj] -= 1   # 平局不计入 total
    net = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if i < j:
                w = wins[i, j]
                t = total[i, j]
                net[i, j] = w - (t - w) if t else 0
            else:
                net[i, j] = -net[j, i]
    net_df = pd.DataFrame(net, index=methods, columns=methods)
    total_df = pd.DataFrame(total, index=methods, columns=methods)
    return net_df, total_df


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def plot_rank_heatmap(rank_mean: pd.DataFrame, rank_count: pd.DataFrame,
                      out_path: Path) -> None:
    _setup_style()
    methods = list(rank_mean.columns)
    metrics = list(rank_mean.index)
    mat = rank_mean.to_numpy(dtype=np.float64)
    vmin, vmax = 1.0, float(len(methods))
    fig, ax = plt.subplots(figsize=(0.7 * len(methods) + 5.0, 0.55 * len(metrics) + 1.8))
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([_method_label(m) for m in methods])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([METRIC_LABEL.get(m, m) for m in metrics])
    for i in range(len(metrics)):
        for j in range(len(methods)):
            v = mat[i, j]
            if np.isfinite(v):
                n = rank_count.iloc[i, j]
                ax.text(j, i, f"{v:.2f}\n(n={int(n)})", ha="center", va="center",
                        fontsize=8)
    ax.set_title("Average rank per metric (lower = better)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="average method rank (1=best)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_win_loss(net: pd.DataFrame, total: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    methods = list(net.columns)
    n = len(methods)
    mat = net.to_numpy(dtype=np.float64)
    lim = float(max(1, int(np.nanmax(np.abs(mat)))))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([_method_label(m) for m in methods])
    ax.set_yticklabels([_method_label(m) for m in methods])
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", color="gray")
                continue
            net_ij = int(mat[i, j])
            t = int(total.iloc[i, j]) if i < j else int(total.iloc[j, i])
            ax.text(j, i, f"{net_ij:+d}\n/{t}", ha="center", va="center", fontsize=8)
    ax.set_title("Win-loss (row A vs col B): net wins / comparable units")
    fig.colorbar(im, ax=ax, shrink=0.8, label="net wins (row vs col)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_platform_boxplots(wide: pd.DataFrame, outdir: Path) -> None:
    """对每个正确性指标画一张 platform_group × method 分组箱线图。"""
    _setup_style()
    for metric in BOXPLOT_METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        groups = [g for g in PLATFORM_ORDER if (wide["platform_group"] == g).any()]
        group_pos = np.arange(len(groups))
        n_method = len(ALL_METHODS)
        width = 0.8 / n_method
        for gi, grp in enumerate(groups):
            sub = wide[wide["platform_group"] == grp]
            for mi, m in enumerate(ALL_METHODS):
                col = f"{m}__{metric}"
                if col not in sub.columns:
                    continue
                vals = sub[col].dropna().astype(float).to_numpy()
                if len(vals) == 0:
                    continue
                pos = group_pos[gi] - 0.4 + (mi + 0.5) * width
                bp = ax.boxplot(vals, positions=[pos], widths=width * 0.85,
                                patch_artist=True, showfliers=False,
                                manage_ticks=False, zorder=2)
                color = _method_color(m)
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                for e in bp["medians"]:
                    e.set_color("black")
                for e in bp["whiskers"] + bp["caps"]:
                    e.set_color(color)
                # 打点显示每个数据集
                jitter = np.random.default_rng(mi * 31 + gi).uniform(-width * 0.3,
                                                                     width * 0.3,
                                                                     size=len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=12,
                           color=color, alpha=0.55, edgecolors="none", zorder=3)
        ax.set_xticks(group_pos)
        ax.set_xticklabels([PLATFORM_LABEL.get(g, g) for g in groups])
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
        ax.set_title(METRIC_LABEL.get(metric, metric) + " by platform")
        handles = [plt.Line2D([0], [0], marker="s", color="w",
                              markerfacecolor=_method_color(m), markersize=9)
                   for m in ALL_METHODS]
        ax.legend(handles, [_method_label(m) for m in ALL_METHODS],
                  loc="best", frameon=False, ncol=len(ALL_METHODS))
        fig.tight_layout()
        fig.savefig(outdir / f"platform_boxplot_{metric}.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)


def plot_fpr(fpr_df: pd.DataFrame, out_path: Path) -> None:
    """坐标置换 FPR 柱状图：mean_fpr ± sd，横参考线 = 名义 α=0.05。"""
    _setup_style()
    if fpr_df.empty:
        return
    datasets = list(dict.fromkeys(fpr_df["dataset"].tolist()))   # 保持出现顺序、去重
    methods = [m for m in ALL_METHODS if m in set(fpr_df["method"])]
    n_meth = len(methods)
    width = 0.8 / n_meth
    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(datasets) * max(n_meth, 1)), 4.2))

    for di, ds in enumerate(datasets):
        sub = fpr_df[fpr_df["dataset"] == ds].set_index("method")
        for mi, m in enumerate(methods):
            if m not in sub.index:
                continue
            r = sub.loc[m]
            pos = di + (mi - (n_meth - 1) / 2) * width
            ax.bar(pos, float(r["mean_fpr"]), width=width * 0.85,
                   color=_method_color(m),
                   yerr=float(r["sd_fpr"]) if pd.notna(r["sd_fpr"]) else None,
                   capsize=3,
                   label=_method_label(m) if di == 0 else None)
    ax.axhline(0.05, ls="--", color="gray", lw=1, label="nominal alpha = 0.05")
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets)
    ax.set_ylabel("mean FPR (coordinate permutation)")
    ax.set_ylim(0, None)
    ax.set_title("Permutation null FPR (lower is better)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="跨数据集对比图（排名热图/win-loss/平台箱线）")
    ap.add_argument("--summary-dir", default=str(ROOT / "results" / "summary"),
                    help="聚合结果目录（默认 results/summary）")
    args = ap.parse_args()

    summary = Path(args.summary_dir)
    outdir = summary / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    wide = load(summary / "metrics_wide.csv", summary / "datasets.csv")
    coverage = pd.read_csv(summary / "method_coverage.csv")

    # 四方法可比口径：仅在 4 方法都跑过（nnSVG 不缺位）的数据集上做 4 方法排名。
    full4 = set(coverage.loc[(coverage[ALL_METHODS] == 1).all(axis=1), "dataset"])
    wide4 = wide[wide["dataset"].isin(full4)]

    # 三方法全量口径：spark/spagcn/spaseg 三个方法在全部 22 个数据集上都有值。
    core3 = ["spark", "spagcn", "spaseg"]

    src.log_message(f"四方法可比数据集数: {len(full4)}；三方法全量数据集数: {len(wide)}")

    # --- 四方法可比口径 ---
    rank_mean, rank_count, _ = compute_rank_matrix(wide4, ALL_METHODS)
    net, total = compute_win_loss(wide4, ALL_METHODS)
    plot_rank_heatmap(rank_mean, rank_count, outdir / "rank_heatmap.png")
    plot_win_loss(net, total, outdir / "win_loss.png")

    # --- 三方法全量口径（消除 nnSVG 幸存者偏差） ---
    rank_mean3, rank_count3, _ = compute_rank_matrix(wide, core3)
    net3, total3 = compute_win_loss(wide, core3)
    plot_rank_heatmap(rank_mean3, rank_count3, outdir / "rank_heatmap_3method.png")
    plot_win_loss(net3, total3, outdir / "win_loss_3method.png")

    plot_platform_boxplots(wide, outdir)

    # --- 坐标置换 FPR（若有） ---
    fpr_path = summary / "fpr.csv"
    if fpr_path.exists():
        fpr_df = pd.read_csv(fpr_path)
        plot_fpr(fpr_df, outdir / "fpr.png")

    # 打印结果供人工检查
    src.log_header("排名汇总（四方法可比口径，平均排名，越小越好）")
    print(rank_mean.round(2).to_string())
    src.log_header("win-loss（四方法可比口径，净胜场 net）")
    print(net.to_string())
    src.log_header("排名汇总（三方法全量口径，平均排名，越小越好）")
    print(rank_mean3.round(2).to_string())
    src.log_header("win-loss（三方法全量口径，净胜场 net）")
    print(net3.to_string())
    src.log_message(f"图已保存到 {outdir.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())