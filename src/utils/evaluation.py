# -*- coding: utf-8 -*-
"""evaluation.py —— 四方法 SVG 检测结果的评价编排层 + 绘图层 + CLI

职责（详见 `paper_material/evaluation_requirements.md`）：
  1. 读取四个方法各自的统一排名 CSV（``SVG_<METHOD>_<sample>_rank.csv``，
     列固定 ``gene,stat,pval,padj,rank``）与 ``runtime.json``；
  2. 读取原始 h5ad（坐标 + 可选类别标注），构建统一的表达矩阵与 k 近邻空间权重；
  3. 调用 `src.utils.metrics` 计算本阶段可落地指标；
  4. 用 matplotlib 产出标准化图表；
  5. 输出 CSV 主表 + ``summary.json``，供跨数据集汇总；
  6. 提供 CLI，被 ``models_benchmark.sh`` 在流水线末端调用。

不做：方法运行、前处理、模拟数据生成。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# matplotlib 内部调用新版 pyparsing 的 deprecated API（parseString/resetCache/
# enablePackrat）会产生大量与本项目无关的 DeprecationWarning，统一忽略。
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        message=r".*deprecated - use .*")

# ---------------------------------------------------------------------------
# 项目路径 & 引入 src 初始化模块（import 无副作用、开销低）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: E402
from src.utils import metrics as M  # noqa: E402

# Cell 风格连续色阶（与 spatial_svg_plots.py 共用 src.CELL_*_COLORS 定义）
CELL_EXPR_CMAP = LinearSegmentedColormap.from_list("cell_green", src.CELL_EXPR_COLORS)
CELL_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "cell_div", src.CELL_DIVERGING_COLORS)

# 各方法产出的统一排名 CSV 文件名前缀（与模型脚本实际写出名一致；
# 注意 spark 目录名为 SPARK_X，但 CSV 前缀为 SPARK）。
RANK_CSV_PREFIX = {
    "spark": "SPARK",
    "nnsvg": "nnSVG",
    "spagcn": "spaGCN",
    "spaseg": "spaSEG",
}

# 随机基因基线（对照）：作为一个伪「方法」出现在对比图/表里，用灰色区分。
RANDOM_BASELINE_KEY = "random"
RANDOM_BASELINE_COLOR = "#8C8C8C"
RANDOM_BASELINE_LABEL = "Random"


def _method_color(m: str) -> str:
    """返回方法颜色；随机基线用灰色。"""
    return src.METHOD_COLORS.get(m, RANDOM_BASELINE_COLOR)


def _method_label(m: str) -> str:
    """返回方法标签；随机基线用 'Random'。"""
    return src.METHOD_LABELS.get(m, RANDOM_BASELINE_LABEL)


# ---------------------------------------------------------------------------
# 读入与对齐
# ---------------------------------------------------------------------------
def read_rank_csv(path: Path) -> pd.DataFrame:
    """读取统一排名 CSV，清洗为 ``gene,stat,pval,padj,rank`` 并按显著性重排序。

    - gene 一律按字符串读入；stat/pval/padj/rank 用 ``pd.to_numeric`` 兜底；
    - Inf/NA 统一成 NaN，NaN padj 排最后；
    - 按 (padj 升序, stat 降序) 重排后重新赋 1..n 的 rank，保证跨方法口径一致。
    """
    df = pd.read_csv(path, dtype={"gene": str})
    for col in ("stat", "pval", "padj", "rank", "effect"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    base_cols = ["gene", "stat", "pval", "padj", "rank"]
    extra = [c for c in ("effect",) if c in df.columns]
    df = df[base_cols + extra]
    df["gene"] = df["gene"].astype(str).str.strip()
    df = df.drop_duplicates(subset="gene")
    df = df.sort_values(
        ["padj", "stat"], ascending=[True, False],
        na_position="last").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def read_runtime(path: Path) -> float:
    """读取 runtime.json 的 wall_seconds；缺失或解析失败记 NaN。"""
    if path is None or not path.exists():
        return np.nan
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("wall_seconds", np.nan))
    except Exception:
        return np.nan


def load_expr_and_coords(run: dict, knn: int):
    """读 h5ad，构建 (log1p 归一化表达矩阵, 空间权重 W, 基因名, 坐标, 标注)。

    返回 ``(expr_mat, W, gene_names, coords, labels)``：
      - expr_mat: (genes x spots) 的 library-size normalize + log1p；
      - W: k 近邻二元对称权重 (n x n)；
      - labels: obs 中的类别标注（无则 None）。
    """
    import anndata as ad
    from scipy import sparse

    h5ad_path = run["h5ad"]
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad 不存在: {h5ad_path}")
    src.log_message(f"读取 h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    src.log_message(f"shape = {adata.shape} (spots x genes)")

    coords_df = src.load_coords(adata, run.get("spatial"))
    keep = [b for b in adata.obs.index if b in coords_df.index]
    coords = coords_df.loc[keep, ["x", "y"]].to_numpy(dtype=np.float64)
    src.log_message(f"对齐坐标后 spots = {len(keep)}")

    # 表达矩阵：优先 raw_count 层（真实 counts），做 library-size normalize + log1p
    if "raw_count" in adata.layers and adata.layers["raw_count"] is not None:
        X = adata.layers["raw_count"]
    else:
        X = adata.X
    if sparse.issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=np.float64)
    spot_idx = [i for i, b in enumerate(adata.obs.index) if b in coords_df.index]
    X = X[spot_idx, :]
    totals = X.sum(axis=1)
    totals[totals == 0] = 1.0
    expr_log = np.log1p(X / totals[:, None] * 1e4)          # spots x genes
    expr_mat = expr_log.T                                    # genes x spots

    gene_names = list(adata.var.index.astype(str))

    # 类别标注（维度 4）
    labels = None
    label_col = None
    for col in ("clusters", "cell_type", "leiden"):
        if col in adata.obs.columns and adata.obs[col].notna().any():
            labels = adata.obs[col].iloc[spot_idx].astype(str).tolist()
            label_col = col
            src.log_message(f"发现类别标注列: obs['{col}']")
            break

    W = M.knn_weights(coords, k=knn)
    del adata
    return expr_mat, W, gene_names, coords, labels, label_col


# ---------------------------------------------------------------------------
# 单方法指标计算
# ---------------------------------------------------------------------------
def method_metrics(method: str, rank_df: pd.DataFrame, runtime: float,
                   expr_mat: np.ndarray, W, gene_names, moran_table: pd.DataFrame,
                   labels, n_clusters: int, top_k_list, n_null: int,
                   seed: int) -> dict:
    """计算单个方法的所有落地指标，返回汇总 dict。"""
    # --- 检出集合（padj < 0.05）---
    sig_genes = rank_df.loc[rank_df["padj"] < 0.05, "gene"].tolist()
    sig_genes = [g for g in sig_genes if g in moran_table.index]
    n_sig = len(sig_genes)
    n_ranked = int(rank_df["rank"].notna().sum())

    sig_morans = (moran_table.loc[sig_genes, "moran_I"].to_numpy(np.float64)
                  if sig_genes else np.asarray([], dtype=np.float64))
    median_moran = float(np.nanmedian(sig_morans)) if len(sig_morans) else np.nan
    median_geary_c = (float(np.nanmedian(moran_table.loc[sig_genes, "geary_C"]))
                      if sig_genes else np.nan)
    median_geary_c_star = (float(np.nanmedian(moran_table.loc[sig_genes, "geary_C_star"]))
                           if sig_genes else np.nan)

    null = M.null_moran_compare(
        sig_morans, moran_table["moran_I"].to_numpy(np.float64),
        n_null=n_null, seed=seed)

    # --- 效应量单调性：-rank vs Moran's I 的 Spearman（正 = 排名越靠前 Moran 越高）---
    common = rank_df[rank_df["gene"].isin(moran_table.index)].copy()
    if len(common) >= 3:
        rho = M.spearman_rho(-common["rank"].to_numpy(np.float64),
                             common["gene"].map(moran_table["moran_I"]).to_numpy(np.float64))
    else:
        rho = np.nan

    # --- 域特异性单调性：-rank vs effect 的 Spearman（针对 SpaGCN 这类 Wilcoxon
    #     域差异方法；effect 列只有该方法 rank CSV 会写出，其它方法恒为 NaN）---
    if "effect" in rank_df.columns and rank_df["effect"].notna().any():
        eff_common = rank_df[rank_df["effect"].notna()].copy()
        eff_rho = (M.spearman_rho(-eff_common["rank"].to_numpy(np.float64),
                                  eff_common["effect"].to_numpy(np.float64))
                   if len(eff_common) >= 3 else np.nan)
    else:
        eff_rho = np.nan

    # --- 输出信息量（静态规则）---
    info_flags = {
        "has_pval": bool(rank_df["pval"].notna().any()),
        "has_fdr": bool(rank_df["padj"].notna().any()),
        "has_effect": method == "nnsvg",      # nnSVG stat = LR_stat（效应量）
        "has_scale": False,
    }

    # --- 1.2 / 1.3 p 值校准（QQ λ + KS 均匀性；用全基因 pval 做校准检查）---
    pvals = rank_df["pval"].to_numpy(np.float64)
    qq = M.qq_plot_lambda(pvals)
    ks_p = M.kstest_uniform(pvals)

    # --- 1.9 信号质量检查（技术假阳性代理）---
    sq = M.signal_quality(expr_mat, gene_names, sig_genes)

    # --- 4.4 解剖结构一致性（仅当有类别标注；用 top-K 基因，K=top_k_list[0]）---
    rd = None
    if labels is not None:
        rd = M.region_discrimination(
            expr_mat.T, gene_names,
            rank_df["gene"].head(top_k_list[0]).tolist(), labels)

    out = {
        "n_sig": n_sig,
        "n_ranked": n_ranked,
        "median_moran_I": median_moran,
        "median_geary_C": median_geary_c,
        "median_geary_C_star": median_geary_c_star,
        "null_median_moran_I": null["median_null"],
        "null_mean_moran_I": null["mean_null"],
        "null_p": null["p_value"],
        "moran_effect_z": null["effect_z"],
        "wall_seconds": runtime,
        "rank_vs_moran_rho": rho,
        "rank_vs_effect_rho": eff_rho,
        "qq_lambda": qq["lambda"],
        "ks_uniform_p": ks_p,
        "frac_low_spots": sq["frac_low_spots"],
        "median_n_spots": sq["median_n_spots"],
        "median_mean_expr": sq["median_mean_expr"],
        "region_eta2": (rd["median_eta2"] if rd is not None else np.nan),
        "region_eta2_mean": (rd["mean_eta2"] if rd is not None else np.nan),
        "info_flags": info_flags,
    }

    # --- 下游价值：top-K SVG -> KMeans -> ARI/NMI（仅当有标注）---
    if labels is not None:
        top_genes = rank_df["gene"].tolist()
        for k in top_k_list:
            r = M.svg_cluster_ari(expr_mat.T, gene_names, top_genes[:k],
                                  labels, n_clusters=n_clusters, seed=seed)
            out[f"ari_top{k}"] = r["ari"]
            out[f"nmi_top{k}"] = r["nmi"]
    return out


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------
def _setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
    })


def plot_pval_hist(rank_dfs: dict, out_path: Path) -> None:
    _setup_style()
    methods = list(rank_dfs)
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        p = rank_dfs[m]["pval"].dropna().to_numpy(np.float64)
        p = p[(p >= 0) & (p <= 1)]
        ax.hist(p, bins=50, color=src.METHOD_COLORS[m], alpha=0.85)
        ax.set_title(src.METHOD_LABELS[m])
        ax.set_xlabel("p-value")
    axes[0][0].set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sig_ratio_curve(rank_dfs: dict, out_path: Path) -> None:
    _setup_style()
    thresholds = np.arange(0.01, 0.101, 0.01)
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, df in rank_dfs.items():
        padj = df["padj"].to_numpy(np.float64)
        n = int(df["padj"].notna().sum())
        ratios = [float((padj < t).sum()) / n if n else 0.0 for t in thresholds]
        ax.plot(thresholds, ratios, marker="o", ms=4,
                color=src.METHOD_COLORS[m], label=src.METHOD_LABELS[m])
    ax.set_xlabel("FDR threshold")
    ax.set_ylabel("Proportion significant")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effect_size_corr(rank_dfs: dict, moran_table: pd.DataFrame,
                          out_path: Path) -> None:
    _setup_style()
    methods = list(rank_dfs)
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5),
                             squeeze=False)
    for ax, m in zip(axes[0], methods):
        df = rank_dfs[m]
        common = df[df["gene"].isin(moran_table.index)]
        x = common["rank"].to_numpy(np.float64)
        y = common["gene"].map(moran_table["moran_I"]).to_numpy(np.float64)
        ax.scatter(x, y, s=6, alpha=0.4, color=src.METHOD_COLORS[m])
        ax.set_title(src.METHOD_LABELS[m])
        ax.set_xlabel("Rank")
    axes[0][0].set_ylabel("Moran's I")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _annotated_heatmap(ax, mat: pd.DataFrame, vmin: float, vmax: float,
                       cmap, title: str) -> None:
    im = ax.imshow(mat.to_numpy(dtype=np.float64), vmin=vmin, vmax=vmax,
                   cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45)
    ax.set_yticklabels(mat.index)
    ax.set_title(title)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            v = mat.iloc[i, j]
            ax.text(j, i, f"{v:.2f}" if np.isfinite(v) else "nan",
                    ha="center", va="center", fontsize=8)
    fig = ax.figure
    fig.colorbar(im, ax=ax, shrink=0.8)


def plot_rank_consensus(spearman_mat: pd.DataFrame, jaccard_mat: pd.DataFrame,
                        jaccard_k: int, out_path: Path) -> None:
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    _annotated_heatmap(axes[0], spearman_mat, -1.0, 1.0, CELL_DIVERGING_CMAP,
                       "Spearman ρ")
    _annotated_heatmap(axes[1], jaccard_mat, 0.0, 1.0, CELL_EXPR_CMAP,
                       f"Top-{jaccard_k} Jaccard")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_n_quality(n_sig: dict, median_moran: dict, null_median: float,
                   out_path: Path, random_baseline: dict = None,
                   random_k: int = None) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    for m in n_sig:
        x = np.log10(max(n_sig[m], 1))
        y = median_moran[m]
        ax.scatter(x, y, s=70, color=_method_color(m))
        ax.annotate(_method_label(m), (x, y),
                    textcoords="offset points", xytext=(6, 2), fontsize=9)
    if np.isfinite(null_median):
        ax.axhline(null_median, ls="--", color="gray", lw=1, label="null median")
    if random_baseline is not None:
        x = np.log10(max(int(random_k or 1), 1))
        y = random_baseline.get("median_moran_I", np.nan)
        if np.isfinite(y):
            ax.scatter(x, y, s=90, color=RANDOM_BASELINE_COLOR, marker="X",
                       zorder=5, label="Random")
    ax.set_xlabel("log10(# detected)")
    ax.set_ylabel("Median Moran's I")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ari_curve(ari_df: pd.DataFrame, out_path: Path) -> None:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    for m, sub in ari_df.groupby("method"):
        ax.plot(sub["top_k"], sub["ari"], marker="o", ms=4,
                color=_method_color(m), label=_method_label(m))
    ax.set_xlabel("Top-K")
    ax.set_ylabel("ARI")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ari_nmi_curve(ari_df: pd.DataFrame, out_path: Path) -> None:
    """下游特征效率曲线：ARI 与 NMI 随 Top-K 变化（双面板折线图，含随机基线）。"""
    _setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, metric, label in ((axes[0], "ari", "ARI"),
                              (axes[1], "nmi", "NMI")):
        for m, sub in ari_df.groupby("method"):
            ls = "--" if m == RANDOM_BASELINE_KEY else "-"
            ax.plot(sub["top_k"], sub[metric], marker="o", ms=4, ls=ls,
                    color=_method_color(m), label=_method_label(m))
        ax.set_xlabel("Top-K")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_qq(rank_dfs: dict, out_path: Path) -> None:
    """1.2 QQ 图：每方法全基因 p 值的 -log10 分位数 vs 均匀理论分位数。"""
    _setup_style()
    methods = list(rank_dfs)
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.5),
                             sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        p = rank_dfs[m]["pval"].dropna().to_numpy(np.float64)
        p = p[(p > 0) & (p <= 1)]
        if len(p) == 0:
            ax.set_title(src.METHOD_LABELS[m])
            continue
        p = np.sort(p)
        n = len(p)
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        observed = -np.log10(p)
        lim = max(float(np.nanmax(expected)), float(np.nanmax(observed))) * 1.05
        ax.scatter(expected, observed, s=6, alpha=0.5,
                   color=src.METHOD_COLORS[m])
        ax.plot([0, lim], [0, lim], ls="--", color="gray", lw=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(src.METHOD_LABELS[m])
    axes[0][0].set_xlabel("Expected -log10(p)")
    axes[0][0].set_ylabel("Observed -log10(p)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _scalar_value(method_results: dict, m: str, key: str, top_k: int) -> float:
    """从 method_results 取一个标量指标值。"""
    return method_results[m].get(key, np.nan)


def plot_scalar_metrics(method_results: dict, methods: list, top_k: int,
                        out_path: Path, baseline: dict = None) -> None:
    """标量指标分组柱状图（每指标一个子图，x=方法，一方法一色，含随机基线）。"""
    _setup_style()
    all_methods = list(methods)
    results = dict(method_results)
    if baseline is not None:
        all_methods = all_methods + [RANDOM_BASELINE_KEY]
        results[RANDOM_BASELINE_KEY] = baseline
    panels = [
        ("median_moran_I", "Median Moran's I"),
        ("median_geary_C_star", "Median Geary C*"),
        ("moran_effect_z", "Moran effect Z"),
        ("rank_vs_moran_rho", "Spearman ρ (rank↔Moran)"),
        (f"ari_top{top_k}", f"ARI @ top-{top_k}"),
        (f"nmi_top{top_k}", f"NMI @ top-{top_k}"),
        ("region_eta2", "Region η²"),
        ("median_n_spots", "Median expr. spots"),
    ]
    n = len(panels)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 2.6),
                             squeeze=False)
    x = np.arange(len(all_methods))
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = [_scalar_value(results, m, key, top_k) for m in all_methods]
        ax.bar(x, vals, color=[_method_color(m) for m in all_methods])
        ax.set_xticks(x)
        ax.set_xticklabels([_method_label(m) for m in all_methods],
                           rotation=30, ha="right")
        ax.set_title(title)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _minmax_col(col: np.ndarray) -> np.ndarray:
    """按列 min-max 归一化到 [0,1]（全 NaN / 全相等时返回 0）。"""
    col = np.asarray(col, dtype=np.float64)
    lo, hi = np.nanmin(col), np.nanmax(col)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.zeros_like(col)
    return (col - lo) / (hi - lo)


def plot_radar(method_results: dict, methods: list, top_k: int,
               out_path: Path, baseline: dict = None) -> None:
    """雷达图：把各标量指标按方法 min-max 归一化后叠加对比（越大越靠外，含随机基线）。"""
    _setup_style()
    all_methods = list(methods)
    results = dict(method_results)
    if baseline is not None:
        all_methods = all_methods + [RANDOM_BASELINE_KEY]
        results[RANDOM_BASELINE_KEY] = baseline
    axes_def = [
        ("Median Moran's I", "median_moran_I", 1.0),
        ("Median Geary C*", "median_geary_C_star", 1.0),
        ("Moran effect Z", "moran_effect_z", 1.0),
        ("Spearman ρ", "rank_vs_moran_rho", 1.0),
        (f"ARI@top{top_k}", f"ari_top{top_k}", 1.0),
        (f"NMI@top{top_k}", f"nmi_top{top_k}", 1.0),
        ("Region η²", "region_eta2", 1.0),
        ("Median expr. spots", "median_n_spots", 1.0),
    ]
    raw = np.zeros((len(all_methods), len(axes_def)))
    for i, m in enumerate(all_methods):
        for j, (_, key, sign) in enumerate(axes_def):
            v = _scalar_value(results, m, key, top_k)
            if sign < 0 and np.isfinite(v):
                v = 1.0 - v
            raw[i, j] = v
    norm = np.column_stack([_minmax_col(raw[:, j])
                            for j in range(len(axes_def))])

    angles = np.linspace(0, 2 * np.pi, len(axes_def), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for i, m in enumerate(all_methods):
        vals = norm[i].tolist() + [norm[i, 0]]
        if m == RANDOM_BASELINE_KEY:
            ax.plot(angles, vals, color=_method_color(m), lw=2, ls="--",
                    label=_method_label(m))
            ax.fill(angles, vals, color=_method_color(m), alpha=0.06)
        else:
            ax.plot(angles, vals, color=_method_color(m), lw=2,
                    label=_method_label(m))
            ax.fill(angles, vals, color=_method_color(m), alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d[0] for d in axes_def], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12), frameon=False)
    fig.suptitle("Normalized comparison (fraction of best method)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主编排
# ---------------------------------------------------------------------------
def run_evaluation(args) -> int:
    methods = _parse_methods(args.methods)
    top_k_list = [int(x) for x in args.top_k_list.split(",") if x.strip()]

    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad,
                          spatial=args.spatial, outdir=args.outdir,
                          sample=args.sample, methods=methods)
    outdir = run["outdir"]
    sample = run["sample"]

    # --- 读入四个方法的统一排名 CSV 与 runtime.json ---
    rank_dfs, runtimes = {}, {}
    missing = []
    for m in run["methods"]:
        subdir = run["method_dirs"][m]
        csv_path = subdir / f"SVG_{RANK_CSV_PREFIX[m]}_{sample}_rank.csv"
        if csv_path.exists():
            try:
                rank_dfs[m] = read_rank_csv(csv_path)
                runtimes[m] = read_runtime(subdir / "runtime.json")
            except Exception as e:
                src.log_message(f"读入 {csv_path} 失败: {e}")
                missing.append(m)
        else:
            missing.append(m)

    if not rank_dfs:
        src.log_message(f"未找到任何统一排名 CSV（期望 *_rank.csv），方法: {missing}")
        return 1
    if missing:
        src.log_message(f"缺失排名 CSV 的方法将被跳过: {missing}")

    methods = [m for m in run["methods"] if m in rank_dfs]
    src.log_message(f"读到排名 CSV 的方法: "
                    f"{[src.METHOD_LABELS[m] for m in methods]}")

    # --- 构建统一表达矩阵与空间权重 ---
    expr_mat, W, gene_names, coords, labels, label_col = load_expr_and_coords(
        run, args.knn)
    moran_table = M.moran_geary_table(expr_mat, gene_names, W)
    has_labels = labels is not None
    n_clusters = int(len(set(labels))) if has_labels else 0

    # --- 逐方法计算指标 ---
    method_results = {}
    for m in methods:
        src.log_message(f"计算 {src.METHOD_LABELS[m]} 指标 ...")
        method_results[m] = method_metrics(
            m, rank_dfs[m], runtimes.get(m, np.nan), expr_mat, W, gene_names,
            moran_table, labels, n_clusters, top_k_list, args.n_null, args.seed)

    # --- 随机基因基线（对照）---
    baseline = M.random_gene_baseline(
        expr_mat, gene_names, labels, moran_table, top_k_list,
        n_clusters=n_clusters, seed=args.seed,
        n_draws=args.n_baseline_draws)

    # --- 一致性（维度 3.5）---
    all_genes = gene_names
    aligned = {m: (df.set_index("gene")["rank"].reindex(all_genes))
               for m, df in rank_dfs.items()}
    spearman_mat = pd.DataFrame(index=methods, columns=methods, dtype=np.float64)
    for a in methods:
        for b in methods:
            spearman_mat.loc[a, b] = M.spearman_rho(aligned[a], aligned[b])

    jaccard_k = top_k_list[0]
    jaccard_mat = pd.DataFrame(index=methods, columns=methods, dtype=np.float64)
    for a in methods:
        for b in methods:
            jaccard_mat.loc[a, b] = M.top_k_jaccard(
                rank_dfs[a]["gene"].tolist(), rank_dfs[b]["gene"].tolist(),
                k=jaccard_k)

    # Kendall's W（仅用各方法都出现的基因，且在共同基因子集上重新局部排名）
    common_genes = all_genes
    for m in methods:
        common_genes = [g for g in common_genes if g in set(rank_dfs[m]["gene"])]
    if len(common_genes) >= 2 and len(methods) >= 2:
        rank_matrix = np.array([
            pd.Series([float(aligned[m].loc[g]) for g in common_genes]
                      ).rank().to_numpy(np.float64)
            for m in methods
        ])
        kendall_w = M.kendall_w(rank_matrix)
    else:
        kendall_w = np.nan

    consensus_top_k = 100 if 100 in top_k_list else top_k_list[0]
    # 共识口径随方法数自适应：4 方法→≥3，3 方法→≥2（避免变成“全交集”口径漂移）
    consensus_min_methods = max(2, len(methods) - 1)
    consensus = M.consensus_genes(rank_dfs, min_methods=consensus_min_methods,
                                  top_k=consensus_top_k)

    # --- 写 tables / figures / summary.json ---
    eval_dir = outdir / "eval"
    tables_dir = eval_dir / "tables"
    figures_dir = eval_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1) metrics_main.csv（长表）
    main_rows = []
    for m in methods:
        r = method_results[m]
        for metric, val in r.items():
            if metric == "info_flags":
                continue
            main_rows.append({"method": m, "metric": metric, "value": val})
    pd.DataFrame(main_rows).to_csv(tables_dir / "metrics_main.csv", index=False)

    # 2) rank_consensus.csv
    cons_rows = []
    for a in methods:
        for b in methods:
            cons_rows.append({"comparison": "spearman", "k": np.nan,
                              "method_a": a, "method_b": b,
                              "value": spearman_mat.loc[a, b]})
            cons_rows.append({"comparison": "jaccard", "k": jaccard_k,
                              "method_a": a, "method_b": b,
                              "value": jaccard_mat.loc[a, b]})
    pd.DataFrame(cons_rows).to_csv(tables_dir / "rank_consensus.csv", index=False)

    # 3) moran_geary.csv（每方法检出集合汇总 + 置换 p）
    moran_rows = [{
        "method": m,
        "n_sig": method_results[m]["n_sig"],
        "median_moran_I": method_results[m]["median_moran_I"],
        "median_geary_C": method_results[m]["median_geary_C"],
        "median_geary_C_star": method_results[m]["median_geary_C_star"],
        "null_median_moran_I": method_results[m]["null_median_moran_I"],
        "null_mean_moran_I": method_results[m]["null_mean_moran_I"],
        "null_p": method_results[m]["null_p"],
        "moran_effect_z": method_results[m]["moran_effect_z"],
    } for m in methods]
    pd.DataFrame(moran_rows).to_csv(tables_dir / "moran_geary.csv", index=False)

    # 4) ari_curve.csv（仅当有标注；含随机基线）
    ari_rows = []
    if has_labels:
        for m in methods:
            for k in top_k_list:
                ari_rows.append({
                    "method": m, "top_k": k,
                    "ari": method_results[m].get(f"ari_top{k}", np.nan),
                    "nmi": method_results[m].get(f"nmi_top{k}", np.nan),
                })
        for k in top_k_list:
            ari_rows.append({
                "method": RANDOM_BASELINE_KEY, "top_k": k,
                "ari": baseline.get(f"ari_top{k}", np.nan),
                "nmi": baseline.get(f"nmi_top{k}", np.nan),
            })
    pd.DataFrame(ari_rows).to_csv(tables_dir / "ari_curve.csv", index=False)

    # 5) pval_calibration.csv（1.2 QQ λ + 1.3 KS 均匀性）
    calib_rows = [{
        "method": m,
        "qq_lambda": method_results[m].get("qq_lambda", np.nan),
        "ks_uniform_p": method_results[m].get("ks_uniform_p", np.nan),
    } for m in methods]
    pd.DataFrame(calib_rows).to_csv(tables_dir / "pval_calibration.csv", index=False)

    # 6) signal_quality.csv（1.9 信号质量检查）
    sq_rows = [{
        "method": m,
        "n_sig": method_results[m].get("n_sig", np.nan),
        "frac_low_spots": method_results[m].get("frac_low_spots", np.nan),
        "median_n_spots": method_results[m].get("median_n_spots", np.nan),
        "median_mean_expr": method_results[m].get("median_mean_expr", np.nan),
    } for m in methods]
    pd.DataFrame(sq_rows).to_csv(tables_dir / "signal_quality.csv", index=False)

    # 7) region_discrimination.csv（4.4 解剖结构一致性 η²）
    rd_rows = [{
        "method": m,
        "median_eta2": method_results[m].get("region_eta2", np.nan),
        "mean_eta2": method_results[m].get("region_eta2_mean", np.nan),
    } for m in methods]
    pd.DataFrame(rd_rows).to_csv(tables_dir / "region_discrimination.csv", index=False)

    # 8) random_baseline.csv（随机基因基线对照，长表 metric/value）
    rand_rows = [{"metric": k, "value": v} for k, v in baseline.items()]
    pd.DataFrame(rand_rows).to_csv(tables_dir / "random_baseline.csv", index=False)

    # --- 绘图（--no-figures 时跳过）---
    if not args.no_figures:
        src.log_message("生成图表 ...")
        try:
            plot_pval_hist(rank_dfs, figures_dir / "pval_hist.png")
            plot_sig_ratio_curve(rank_dfs, figures_dir / "sig_ratio_curve.png")
            plot_qq(rank_dfs, figures_dir / "qq.png")
            plot_effect_size_corr(rank_dfs, moran_table,
                                  figures_dir / "effect_size_corr.png")
            if len(methods) >= 2:
                plot_rank_consensus(spearman_mat, jaccard_mat, jaccard_k,
                                    figures_dir / "rank_consensus.png")
            null_median = float(np.nanmedian(
                [method_results[m]["null_median_moran_I"] for m in methods]))
            plot_n_quality({m: method_results[m]["n_sig"] for m in methods},
                           {m: method_results[m]["median_moran_I"] for m in methods},
                           null_median, figures_dir / "n_quality.png",
                           random_baseline=baseline, random_k=top_k_list[0])
            plot_scalar_metrics(method_results, methods, top_k_list[0],
                                figures_dir / "scalar_comparison.png",
                                baseline=baseline)
            if len(methods) >= 2:
                plot_radar(method_results, methods, top_k_list[0],
                           figures_dir / "radar.png", baseline=baseline)
            if has_labels and ari_rows:
                ari_df = pd.DataFrame(ari_rows)
                plot_ari_curve(ari_df, figures_dir / "ari_curve.png")
                plot_ari_nmi_curve(ari_df, figures_dir / "ari_nmi_curve.png")
        except Exception as e:
            src.log_message(f"绘图失败（不影响表与 summary）: {e}")

    # --- summary.json ---
    summary = {
        "sample": sample,
        "dataset": run["dataset"],
        "n_genes": int(len(gene_names)),
        "n_spots": int(expr_mat.shape[1]),
        "has_labels": has_labels,
        "label_col": label_col if has_labels else None,
        "methods": {
            m: {
                "n_sig": method_results[m]["n_sig"],
                "n_ranked": method_results[m]["n_ranked"],
                "median_moran_I": method_results[m]["median_moran_I"],
                "null_p": method_results[m]["null_p"],
                "median_geary_C_star": method_results[m]["median_geary_C_star"],
                "moran_effect_z": method_results[m]["moran_effect_z"],
                "wall_seconds": method_results[m]["wall_seconds"],
                "rank_vs_moran_rho": method_results[m]["rank_vs_moran_rho"],
                "rank_vs_effect_rho": method_results[m]["rank_vs_effect_rho"],
                "qq_lambda": method_results[m]["qq_lambda"],
                "ks_uniform_p": method_results[m]["ks_uniform_p"],
                "frac_low_spots": method_results[m]["frac_low_spots"],
                "median_n_spots": method_results[m]["median_n_spots"],
                "median_mean_expr": method_results[m]["median_mean_expr"],
                "region_eta2": method_results[m]["region_eta2"],
                "region_eta2_mean": method_results[m]["region_eta2_mean"],
                "info_flags": method_results[m]["info_flags"],
            } for m in methods
        },
        "random_baseline": baseline,
        "consistency": {
            "spearman_matrix": {
                a: {b: spearman_mat.loc[a, b] for b in methods} for a in methods
            },
            "jaccard_matrix": {
                a: {b: jaccard_mat.loc[a, b] for b in methods} for a in methods
            },
            "kendall_w": kendall_w,
            f"consensus_genes_top{consensus_top_k}": consensus,
        },
    }
    if has_labels:
        for m in methods:
            for k in top_k_list:
                summary["methods"][m][f"ari_top{k}"] = method_results[m].get(
                    f"ari_top{k}", np.nan)
                summary["methods"][m][f"nmi_top{k}"] = method_results[m].get(
                    f"nmi_top{k}", np.nan)

    with (eval_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, ensure_ascii=False)

    # --- 打印汇总到 stdout（供 pipeline 日志）---
    src.log_header("evaluation 汇总")
    src.log_message(f"sample={sample}  genes={len(gene_names)}  spots={expr_mat.shape[1]}"
                    f"  labels={has_labels}")
    for m in methods:
        r = method_results[m]
        src.log_message(f"[{src.METHOD_LABELS[m]:>7}] n_sig={r['n_sig']:>4} "
                        f"median_moran={r['median_moran_I']:.3f} "
                        f"null_p={r['null_p']:.3f} wall={r['wall_seconds']:.1f}s")
    src.log_message(f"eval 产物目录: {eval_dir}")
    return 0


def _sanitize_json(o):
    """递归把 numpy 标量 / 非有限浮点转成 JSON 安全的 Python 原生类型。"""
    if isinstance(o, dict):
        return {k: _sanitize_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_sanitize_json(v) for v in o.tolist()]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    return o


def _parse_methods(methods_arg) -> list:
    if not methods_arg:
        return None
    return [m for m in re.split(r"[,\s]+", methods_arg.strip()) if m]


def main():
    ap = argparse.ArgumentParser(description="四方法 SVG 检测评价（编排 + 绘图 + 汇总）")
    ap.add_argument("--dataset", default=None, help="dataset key")
    ap.add_argument("--h5ad", default=None, help="输入 h5ad（绝对或相对项目根）")
    ap.add_argument("--spatial", default=None, help="可选坐标文件")
    ap.add_argument("--outdir", default=None, help="输出根目录")
    ap.add_argument("--sample", default=None, help="样本标签（默认 dataset）")
    ap.add_argument("--methods", default=None,
                    help="方法子集（逗号或空格分隔，默认全部）")
    ap.add_argument("--knn", type=int, default=6, help="Moran's I 近邻数（默认 6）")
    ap.add_argument("--n-null", type=int, default=200,
                    help="随机对照置换次数（默认 200）")
    ap.add_argument("--n-baseline-draws", type=int, default=10,
                    help="随机基因基线抽样次数（默认 10）")
    ap.add_argument("--top-k-list", default="100,500,1000",
                    help="一致性/下游用的 top-K 列表（逗号分隔）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    ap.add_argument("--no-figures", action="store_true", help="跳过绘图")
    args = ap.parse_args()

    sys.exit(run_evaluation(args))


if __name__ == "__main__":
    main()
