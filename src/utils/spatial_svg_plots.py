# -*- coding: utf-8 -*-
"""spatial_svg_plots.py —— 四方法 Top SVG 的空间分布可视化（非交互 PNG）

与 `evaluation.py` 产出的统计图互补：本模块回答"每个方法检出的 Top SVG 在切片上
长什么样、彼此有何不同"，突出不同 SVG 检测方法在空间模式上的差异。

设计要点：
  1. 统一数据源：读原始 h5ad（复用 evaluation.load_expr_and_coords 得到
     library-size normalize + log1p 表达与坐标），再读各方法统一排名 CSV
     ``SVG_<METHOD>_<sample>_rank.csv`` 取 Top-N 基因。四方法坐标/表达口径一致。
  2. 同基因同色：``gene_color(gene)`` 基于基因名做确定性 hash -> HSV，全模块所有图
     复用，保证同一 SVG 在任意方法、任意图里颜色一致。
  3. 非交互：matplotlib Agg 后端，纯 PNG，适合 Linux/HPC。
  4. 2D 数据；3D（Stereo-seq/Slide-seq，仅 SPARK-X）暂不绘制（留待后续按切片投影）。

产出（默认写到 ``<outdir>/eval/figures/spatial/``）：
  top_expr_<sample>.png       每方法 Top-N 基因空间表达小图矩阵（Cell 绿色系）
  dominant_gene_<sample>.png  每方法 Top-N 基因"主导基因"空间图（同基因同色）
  cross_method_<sample>.png   跨方法对照矩阵（行=基因，列=方法）
  pattern_gallery_<sample>_<method>.png  raw/平滑/残差 三连图（聚集/梯度/边沿）
  pattern_classify_<sample>.png Moran's I vs 空间梯度 R² 散点
  overlap_<sample>.png        Top-N 基因集合 UpSet 重叠图
  unique_genes_<sample>.png   各方法独有 Top 基因的空间分布

用法（项目根，用 envs/spatial 的 python）：
    python src/utils/spatial_svg_plots.py --dataset DLPFC_151507
    python src/utils/spatial_svg_plots.py --dataset DLPFC_151507 --top-n 10
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# 项目路径 & 引入 src 初始化模块
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: E402
from src.utils import metrics as M  # noqa: E402
from src.utils.evaluation import (  # noqa: E402
    RANK_CSV_PREFIX,
    load_expr_and_coords,
    read_rank_csv,
)

# ---------------------------------------------------------------------------
# 常量：Cell 风格连续色阶（低饱和、色盲友好，取代 matplotlib 默认 viridis/RdBu）
# ---------------------------------------------------------------------------
CMAP_CONT = LinearSegmentedColormap.from_list("cell_green", src.CELL_EXPR_COLORS)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "cell_div", src.CELL_DIVERGING_COLORS)
BACKGROUND_GRAY = (0.92, 0.92, 0.92)


# ---------------------------------------------------------------------------
# 同基因同色：确定性 hash -> HSV
# ---------------------------------------------------------------------------
def gene_color(gene: str):
    """返回某个基因的稳定颜色（RGB 元组，0~1）。同名基因永远同色。

    采用 Cell 风格的"低饱和" HSV（降低饱和度与明度），避免默认高饱和色过于刺眼；
    360 档色相保证任意少量基因几乎不撞色，且跨方法/跨图颜色一致。
    """
    h = hashlib.sha1(str(gene).encode("utf-8")).hexdigest()
    hue = int(h[:6], 16) % 360 / 360.0
    import colorsys

    return colorsys.hsv_to_rgb(hue, 0.48, 0.88)


# ---------------------------------------------------------------------------
# 数值与绘图辅助
# ---------------------------------------------------------------------------
def _robust_scale(expr: np.ndarray) -> np.ndarray:
    """按 1%~99% 分位数把表达缩放到 [0,1]（恒定或含 NaN 时返回 0）。"""
    e = np.asarray(expr, dtype=np.float64)
    if e.size == 0:
        return np.zeros_like(e)
    lo, hi = np.nanpercentile(e, 1), np.nanpercentile(e, 99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(e)
    return np.clip((e - lo) / (hi - lo), 0.0, 1.0)


def _symmetric_scale(expr: np.ndarray) -> tuple:
    """返回 (归一化残差, vmax)：以 |max| 对称中心 0 的 diverging 归一化。"""
    e = np.asarray(expr, dtype=np.float64)
    vmax = float(np.nanmax(np.abs(e))) if e.size else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    return np.clip(e / vmax, -1.0, 1.0), vmax


def _draw_spatial(ax, coords: np.ndarray, values, cmap=CMAP_CONT,
                  vmin=None, vmax=None, s: float = 8.0):
    """在 ax 上画一张 2D 空间散点表达图，返回 scatter（供 colorbar）。"""
    v = np.asarray(values, dtype=np.float64)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=v, cmap=cmap,
        vmin=vmin, vmax=vmax, s=s, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc


def _draw_dominant(ax, coords: np.ndarray, colors, mask=None, s: float = 8.0):
    """画按基因着色的空间散点（colors 为每个 spot 的颜色，可为 'none' 字符串）。"""
    if mask is not None:
        colors = [c if m else BACKGROUND_GRAY for c, m in zip(colors, mask)]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=s,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _blank_cell(ax, text: str = ""):
    """空白格子（基因缺失 / 未进 Top-N 时占位）。"""
    ax.set_facecolor(BACKGROUND_GRAY)
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9,
            transform=ax.transAxes, color="0.3")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color("0.8")


def _subsample(coords, expr_mat, max_spots, seed: int):
    """spot 过多时随机下采样，避免散点图在超大平台上卡死。"""
    n = coords.shape[0]
    if n <= max_spots:
        return coords, expr_mat
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_spots, replace=False)
    src.log_message(f"spot 数 {n} 超过 --max-spots={max_spots}，"
                    f"下采样到 {max_spots} 个 spot 用于绘图")
    return coords[idx], expr_mat[:, idx]


# ---------------------------------------------------------------------------
# 数据装载
# ---------------------------------------------------------------------------
def _parse_methods(methods_arg) -> list:
    if not methods_arg:
        return None
    return [m for m in methods_arg.replace(",", " ").split() if m]


def _load_data(run: dict, methods: list, args):
    """读 rank CSV 与统一表达/坐标，返回 (rank_dfs, top_genes, expr_mat, W, coords, gene_names)。"""
    sample = run["sample"]
    rank_dfs, top_genes = {}, {}
    for m in methods:
        csv_path = run["method_dirs"][m] / f"SVG_{RANK_CSV_PREFIX[m]}_{sample}_rank.csv"
        if csv_path.exists():
            df = read_rank_csv(csv_path)
            rank_dfs[m] = df
            top_genes[m] = df["gene"].head(args.top_n).tolist()
        else:
            src.log_message(f"缺少排名 CSV，跳过 {src.METHOD_LABELS[m]}: {csv_path}")

    methods = [m for m in methods if m in rank_dfs]
    if not methods:
        return None, None, None, None, None, None

    expr_mat, W, gene_names, coords, _labels, _label_col = load_expr_and_coords(
        run, args.knn)

    # 下采样后需用新坐标重建 W（Moran / 平滑都依赖 W）
    coords, expr_mat = _subsample(coords, expr_mat, args.max_spots, args.seed)
    W = M.knn_weights(coords, k=args.knn)

    return rank_dfs, top_genes, expr_mat, W, coords, gene_names


def _gene_index(gene_names: list) -> dict:
    return {g: i for i, g in enumerate(gene_names)}


# ---------------------------------------------------------------------------
# 图 1：每方法 Top-N 空间表达小图矩阵（基线）
# ---------------------------------------------------------------------------
def plot_top_expr(methods, top_genes, expr_mat, coords, gene_names,
                  out_path: Path, s: float):
    gidx = _gene_index(gene_names)
    top_n = max((len(top_genes[m]) for m in methods), default=0)
    if top_n == 0:
        return
    fig, axes = plt.subplots(len(methods), top_n,
                             figsize=(top_n * 2.1, len(methods) * 2.1),
                             squeeze=False)
    sc = None
    for r, m in enumerate(methods):
        genes = top_genes[m][:top_n]
        for c in range(top_n):
            ax = axes[r, c]
            if c < len(genes) and genes[c] in gidx:
                expr = expr_mat[gidx[genes[c]]]
                sc = _draw_spatial(ax, coords, _robust_scale(expr),
                                   CMAP_CONT, s=s)
                ax.set_title(genes[c], fontsize=9, style="italic",
                             color=gene_color(genes[c]))
            else:
                _blank_cell(ax, "N/A")
        axes[r, 0].set_ylabel(src.METHOD_LABELS[m], fontsize=10)
    if sc is not None:
        fig.colorbar(sc, ax=axes, shrink=0.9, label="scaled log1p expr")
    fig.suptitle("Top SVG spatial expression (per method)", fontsize=12)
    fig.subplots_adjust(top=0.92, bottom=0.06, left=0.08, right=0.92,
                        wspace=0.12, hspace=0.30)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成 Top 空间表达矩阵: {out_path}")


# ---------------------------------------------------------------------------
# 图 2：主导基因空间图（同基因同色）
# ---------------------------------------------------------------------------
def plot_dominant(methods, top_genes, expr_mat, coords, gene_names,
                  out_path: Path, s: float):
    gidx = _gene_index(gene_names)
    fig, axes = plt.subplots(1, len(methods),
                             figsize=(len(methods) * 3.2, 3.4),
                             squeeze=False)
    legend_genes = []
    for c, m in enumerate(methods):
        ax = axes[0, c]
        genes = [g for g in top_genes[m] if g in gidx]
        for g in genes:
            if g not in legend_genes:
                legend_genes.append(g)
        if not genes:
            _blank_cell(ax, "N/A")
            ax.set_title(src.METHOD_LABELS[m], fontsize=10)
            continue
        mat = np.vstack([expr_mat[gidx[g]] for g in genes])          # genes x spots
        z = (mat - mat.mean(axis=1, keepdims=True)) / \
            (mat.std(axis=1, keepdims=True) + 1e-12)
        dom = np.argmax(z, axis=0)
        max_z = np.max(z, axis=0)
        colors = [gene_color(genes[i]) for i in dom]
        _draw_dominant(ax, coords, colors, mask=(max_z >= 0), s=s)
        ax.set_title(src.METHOD_LABELS[m], fontsize=10)

    # 共享图例放图下方，避免遮挡空间散点（同一基因同一颜色，一个图例即可覆盖所有面板）
    if legend_genes:
        handles = [Patch(facecolor=gene_color(g), label=g) for g in legend_genes]
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02), ncol=min(len(legend_genes), 6),
                   fontsize=7, frameon=False)

    fig.suptitle("Dominant top-SVG per spot (same gene = same color)", fontsize=12)
    fig.subplots_adjust(top=0.86, bottom=0.16, left=0.03, right=0.97,
                        wspace=0.05)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成主导基因空间图: {out_path}")


# ---------------------------------------------------------------------------
# 图 3：跨方法对照矩阵
# ---------------------------------------------------------------------------
def _union_gene_order(top_genes, max_cols: int) -> list:
    """按"被多少方法检出 -> 最小排名"排序的基因并集，最多取 max_cols 个。"""
    from collections import Counter

    cnt = Counter()
    best = {}
    for m, genes in top_genes.items():
        for rank, g in enumerate(genes):
            cnt[g] += 1
            best[g] = min(best.get(g, 1e9), rank)
    genes = list(cnt)
    genes.sort(key=lambda g: (-cnt[g], best[g], g))
    return genes[:max_cols]


def plot_cross_method(methods, top_genes, expr_mat, coords, gene_names,
                      out_path: Path, s: float, max_cols: int):
    gidx = _gene_index(gene_names)
    union = _union_gene_order(top_genes, max_cols)
    if not union:
        return
    fig, axes = plt.subplots(len(union), len(methods),
                             figsize=(len(methods) * 2.4, len(union) * 2.1),
                             squeeze=False)
    for r, g in enumerate(union):
        for c, m in enumerate(methods):
            ax = axes[r, c]
            if g in top_genes[m] and g in gidx:
                _draw_spatial(ax, coords, _robust_scale(expr_mat[gidx[g]]),
                              CMAP_CONT, s=s)
            else:
                _blank_cell(ax, "×")
            if c == 0:
                ax.set_ylabel(g, fontsize=9, style="italic",
                              color=gene_color(g), rotation=0,
                              ha="right", va="center")
        axes[r, 0].yaxis.labelpad = 0
    for c, m in enumerate(methods):
        axes[0, c].set_title(src.METHOD_LABELS[m], fontsize=10)
    fig.suptitle("Cross-method comparison of top SVGs (grey = not in top-N)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成跨方法对照矩阵: {out_path}")


# ---------------------------------------------------------------------------
# 图 4：空间模式三连图 + 分类散点
# ---------------------------------------------------------------------------
def _smooth_expr(expr, W):
    e = np.asarray(expr, dtype=np.float64)
    denom = np.asarray(W.sum(axis=1)).ravel()
    denom[denom == 0] = 1.0
    return np.asarray(W @ e).ravel() / denom


def plot_pattern_gallery(methods, top_genes, expr_mat, W, coords, gene_names,
                         out_dir: Path, sample: str, s: float):
    gidx = _gene_index(gene_names)
    for m in methods:
        genes = [g for g in top_genes[m] if g in gidx]
        if not genes:
            continue
        fig, axes = plt.subplots(len(genes), 3,
                                 figsize=(3 * 2.4, len(genes) * 2.1),
                                 squeeze=False)
        for r, g in enumerate(genes):
            e = expr_mat[gidx[g]]
            smooth = _smooth_expr(e, W)
            resid = e - smooth
            _draw_spatial(axes[r, 0], coords, _robust_scale(e), CMAP_CONT, s=s)
            _draw_spatial(axes[r, 1], coords, _robust_scale(smooth), CMAP_CONT, s=s)
            rv, vmax = _symmetric_scale(resid)
            _draw_spatial(axes[r, 2], coords, rv, CMAP_DIVERGING,
                          vmin=-vmax, vmax=vmax, s=s)
            axes[r, 0].set_ylabel(g, fontsize=9, style="italic",
                                  color=gene_color(g), rotation=0,
                                  ha="right", va="center")
        for c, title in enumerate(["raw", "smoothed", "residual (edge)"]):
            axes[0, c].set_title(title, fontsize=10)
        fig.suptitle(f"Spatial pattern gallery — {src.METHOD_LABELS[m]}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out = out_dir / f"pattern_gallery_{sample}_{m}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        src.log_message(f"已生成空间模式三连图: {out}")


def _gradient_r2(expr, coords):
    from sklearn.linear_model import LinearRegression

    e = np.asarray(expr, dtype=np.float64)
    if e.std() == 0.0 or len(e) < 3:
        return 0.0
    return float(LinearRegression().fit(coords, e).score(coords, e))


def plot_pattern_classify(methods, top_genes, expr_mat, W, coords, gene_names,
                          out_path: Path):
    gidx = _gene_index(gene_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    seen = set()
    for m in methods:
        genes = [g for g in top_genes[m] if g in gidx]
        xs, ys, labels = [], [], []
        for g in genes:
            if g in seen:
                continue
            seen.add(g)
            e = expr_mat[gidx[g]]
            xs.append(M.morans_i(e, W))
            ys.append(_gradient_r2(e, coords))
            labels.append(g)
        if xs:
            ax.scatter(xs, ys, s=60, color=src.METHOD_COLORS[m],
                       label=src.METHOD_LABELS[m], zorder=3)
            for x, y, g in zip(xs, ys, labels):
                ax.annotate(g, (x, y), fontsize=7, alpha=0.85,
                            textcoords="offset points", xytext=(4, 2))
    ax.set_xlabel("Moran's I (clustering / spatial autocorrelation)")
    ax.set_ylabel("Spatial gradient R² (smooth trend)")
    ax.axhline(0.5, ls="--", color="gray", lw=1, alpha=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.suptitle("Top SVGs in clustering-gradient space")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成空间模式分类散点: {out_path}")


# ---------------------------------------------------------------------------
# 图 5：集合重叠 UpSet + 独有基因分布
# ---------------------------------------------------------------------------
def plot_overlap(methods, top_genes, out_path: Path):
    sets = {m: set(top_genes[m]) for m in methods}
    labels = list(sets)
    if len(labels) < 2:
        return

    combos = []
    n = len(labels)
    for r in range(1, n + 1):
        for c in combinations(range(n), r):
            inter = set().union(*sets.values())
            for i in c:
                inter &= sets[labels[i]]
            if inter:
                combos.append((c, len(inter)))
    combos.sort(key=lambda t: (-t[1], t[0]))

    fig = plt.figure(figsize=(max(9, 3 + 0.6 * len(combos)), 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 3.0], wspace=0.35)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    for i, m in enumerate(labels):
        ax_left.barh(i, len(sets[m]), color=src.METHOD_COLORS[m], alpha=0.85)
    ax_left.set_yticks(range(n))
    ax_left.set_yticklabels([src.METHOD_LABELS[m] for m in labels])
    ax_left.invert_yaxis()
    ax_left.set_xlabel("set size")
    ax_left.set_title("Top-N sets", fontsize=10)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    xpos = list(range(len(combos)))
    counts = [c[1] for c in combos]
    ax_right.bar(xpos, counts, color="0.55")
    for x, c in zip(xpos, counts):
        ax_right.text(x, c + 0.1, str(c), ha="center", va="bottom", fontsize=8)
    ax_right.set_xticks(xpos)
    ax_right.set_xticklabels(
        [" ∩ ".join(src.METHOD_LABELS[labels[i]] for i in c) for c, _ in combos],
        rotation=45, ha="right", fontsize=8)
    ax_right.set_ylabel("intersection size")
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    fig.suptitle("Top-N SVG set overlap across methods")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成集合重叠图: {out_path}")


def plot_unique_genes(methods, top_genes, expr_mat, coords, gene_names,
                      out_path: Path, s: float):
    gidx = _gene_index(gene_names)
    sets = {m: set(top_genes[m]) for m in methods}
    uniques = {}
    for m in methods:
        others = set().union(*[sets[o] for o in methods if o != m])
        uniques[m] = [g for g in top_genes[m] if g not in others and g in gidx]

    if not any(uniques.values()):
        src.log_message("无方法独有 Top 基因，跳过独有基因分布图")
        return

    max_len = max((len(v) for v in uniques.values()), default=0)
    fig, axes = plt.subplots(len(methods), max_len,
                             figsize=(max_len * 2.1, len(methods) * 2.1),
                             squeeze=False)
    for r, m in enumerate(methods):
        for c in range(max_len):
            ax = axes[r, c]
            if c < len(uniques[m]):
                g = uniques[m][c]
                _draw_spatial(ax, coords, _robust_scale(expr_mat[gidx[g]]),
                              CMAP_CONT, s=s)
                ax.set_title(g, fontsize=9, style="italic", color=gene_color(g))
            else:
                _blank_cell(ax)
        axes[r, 0].set_ylabel(src.METHOD_LABELS[m], fontsize=10)
    fig.suptitle("Unique top SVGs per method (spatial expression)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已生成独有基因分布图: {out_path}")


# ---------------------------------------------------------------------------
# 主编排 + CLI
# ---------------------------------------------------------------------------
def run_spatial_plots(args) -> int:
    methods = _parse_methods(args.methods)
    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad,
                          spatial=args.spatial, outdir=args.outdir,
                          sample=args.sample, methods=methods)
    if int(run.get("dim") or 2) == 3:
        src.log_message("3D 空间 SVG 可视化暂未实现，跳过（仅 SPARK-X 支持 3D）")
        return 0

    methods = run["methods"]
    out_dir = run["outdir"] / "eval" / "figures" / "spatial"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = run["sample"]

    src.log_header(f"空间 SVG 可视化: {sample}")
    loaded = _load_data(run, methods, args)
    if loaded[0] is None:
        src.log_message("未找到任何排名 CSV，无法绘制空间图")
        return 1
    rank_dfs, top_genes, expr_mat, W, coords, gene_names = loaded
    methods = [m for m in methods if m in rank_dfs]
    src.log_message(f"方法: {[src.METHOD_LABELS[m] for m in methods]} "
                    f"| spots={coords.shape[0]} | genes={len(gene_names)}")

    fig_sel = set((args.figures or "all").replace(",", " ").split())
    do = lambda name: ("all" in fig_sel) or (name in fig_sel)
    s = args.point_size

    if do("top_expr"):
        plot_top_expr(methods, top_genes, expr_mat, coords, gene_names,
                      out_dir / f"top_expr_{sample}.png", s)
    if do("dominant"):
        plot_dominant(methods, top_genes, expr_mat, coords, gene_names,
                      out_dir / f"dominant_gene_{sample}.png", s)
    if do("cross_method"):
        plot_cross_method(methods, top_genes, expr_mat, coords, gene_names,
                          out_dir / f"cross_method_{sample}.png", s,
                          args.matrix_max_genes)
    if do("pattern"):
        plot_pattern_gallery(methods, top_genes, expr_mat, W, coords, gene_names,
                             out_dir, sample, s)
        plot_pattern_classify(methods, top_genes, expr_mat, W, coords, gene_names,
                              out_dir / f"pattern_classify_{sample}.png")
    if do("overlap"):
        plot_overlap(methods, top_genes, out_dir / f"overlap_{sample}.png")
        plot_unique_genes(methods, top_genes, expr_mat, coords, gene_names,
                          out_dir / f"unique_genes_{sample}.png", s)

    src.log_message(f"空间可视化产物目录: {out_dir}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="四方法 Top SVG 空间分布可视化（PNG）")
    ap.add_argument("--dataset", default=None, help="dataset key")
    ap.add_argument("--h5ad", default=None, help="输入 h5ad（覆盖 dataset）")
    ap.add_argument("--spatial", default=None, help="可选坐标文件")
    ap.add_argument("--outdir", default=None, help="输出根目录")
    ap.add_argument("--sample", default=None, help="样本标签（默认 dataset）")
    ap.add_argument("--methods", default=None, help="方法子集（逗号分隔，默认全部）")
    ap.add_argument("--top-n", type=int, default=6, help="每方法取前 N 个 SVG（默认 6）")
    ap.add_argument("--knn", type=int, default=6, help="空间近邻数（默认 6）")
    ap.add_argument("--figures", default="all",
                    help="要画的图，逗号分隔：top_expr,dominant,cross_method,pattern,overlap"
                         "（默认 all）")
    ap.add_argument("--max-spots", type=int, default=300000,
                    help="散点图 spot 上限，超过则随机下采样（默认 300000）")
    ap.add_argument("--matrix-max-genes", type=int, default=12,
                    help="跨方法矩阵最多展示的基因列数（默认 12）")
    ap.add_argument("--point-size", type=float, default=8.0,
                    help="散点大小（默认 8）")
    ap.add_argument("--seed", type=int, default=0, help="下采样随机种子（默认 0）")
    args = ap.parse_args()
    sys.exit(run_spatial_plots(args))


if __name__ == "__main__":
    main()
