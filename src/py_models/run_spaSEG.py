# -*- coding: utf-8 -*-
"""
run_spaSEG.py
=============
在 mouse_brain_STARmap 数据上运行 SpaSEG 检测空间可变基因 (SVG)。

流程（对应官方 notebook/SVG 的 SVG 检测 demo，做最简/默认参数设置）：
    1. 读取 h5ad（坐标来自 obsm['spatial']）
    2. 表达预处理（X 已是 log1p+normalize，直接 PCA 作为 SpaSEG 输入特征）
    3. 将连续坐标等比缩放为紧凑整数网格 (array_row/array_col)，
       与官方 add_spot_pos 对 Slide-seqV2/MERFISH 等 cell-level 平台的处理一致
    4. SpaSEG 卷积分割训练 -> 得到空间域标签 SpaSEG_clusters
    5. detect_svg() 逐空间域做 Wilcoxon 差异表达 + 过滤 -> SVG 结果
    6. 保存 SVG_spaSEG_STARmap_Mouse_Brain.csv 并绘制空间展示图

结果输出目录:
    results/local_results/mouse_brain_STARmap/spaSEG/
      - SVG_spaSEG_STARmap_Mouse_Brain.csv
      - domains_spaSEG_STARmap_Mouse_Brain.png
      - SVG_spaSEG_STARmap_Mouse_Brain_<gene>.png

注：本机 torch 为 CPU 版（无 GPU），SpaSEG 默认 iterations=2100 在 CPU 上过慢，
    故采用"最简可跑通"的缩减迭代设置（iterations/pretrain_epochs 见下方常量）。
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0) 常量与路径
# ---------------------------------------------------------------------------
SPASEG_SRC = r"F:\computatinalbiology\env_spatial\SpaSEG_src"      # 官方源码根目录
DATA_H5AD = r"F:\computatinalbiology\data\STARmap\mouse_brain_cortex\mouse_brain_STARmap_processed.h5ad"
OUTDIR = Path(r"F:\computatinalbiology\results\local_results\mouse_brain_STARmap\spaSEG")
SAMPLE_NAME = "STARmap_Mouse_Brain"

# --- SpaSEG 最简参数 ---
COMPONS = 15          # PCA 维数 == input_dim/nChannel/output_dim
ALPHA = 0.4           # sim_weight（分割损失权重，默认值）
BETA = 0.7            # con_weight（边缘连续性损失权重，默认值）
MIN_LABEL = 7         # 域数 <= 该值时提前停止（demo 默认）
PRETRAIN_EPOCHS = 400 # 官方默认（仅重建损失的预训练轮数）
ITERATIONS = 1600     # 官方默认 2100；本机为 CPU 且 ~20 it/s，1600 已足够且能跑通
POSITION_MAX = 500.0  # 坐标等比缩放后最大边的上限（与官方 MERFISH/Slide-seq 一致量级）

TOP_N = 6             # 绘制 Top N 个 SVG 的空间表达图

# 让 SpaSEG 源码可被 import
sys.path.insert(0, SPASEG_SRC)


def print_header_step(i, total, msg):
    print(f"[{i}/{total}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1) 读取 + 预处理
# ---------------------------------------------------------------------------
def load_data():
    print_header_step(1, 6, f"读取 h5ad: {DATA_H5AD}")
    adata = sc.read_h5ad(DATA_H5AD)
    print(f"      shape = {adata.shape} (spots x genes)", flush=True)
    print(f"      X 类型 = {type(adata.X).__name__}, max = {adata.X.max():.3f}", flush=True)
    print(f"      坐标来源 = obsm['spatial'], 形状 = {adata.obsm['spatial'].shape}", flush=True)
    return adata


def preprocess(adata):
    """表达 PCA + 构造 SpaSEG 所需的整数网格坐标 array_row/array_col。"""
    print_header_step(2, 6, "预处理（PCA 特征 + 坐标网格化）")

    # 2.1 表达特征：X 已是 log1p+normalize，直接 PCA 作为 SpaSEG 的输入表达
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=COMPONS, random_state=0)
    else:
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :COMPONS]
    print(f"      X_pca 形状 = {adata.obsm['X_pca'].shape}", flush=True)

    # 2.2 坐标网格化（官方 add_spot_pos 对 cell-level 数据的处理思路）：
    #     把连续坐标平移到 >=0 后等比缩放（最大边 -> POSITION_MAX），再取整为整数网格。
    #     注意：官方约定 array_row 对应 spatial[:,0] (x)，array_col 对应 spatial[:,1] (y)。
    xy = adata.obsm["spatial"].astype(np.float64)
    scale = POSITION_MAX / np.max(xy)
    arr_row = (xy[:, 0] - xy[:, 0].min()) * scale
    arr_col = (xy[:, 1] - xy[:, 1].min()) * scale

    adata.obs["array_row"] = arr_row.astype(int)
    adata.obs["array_col"] = arr_col.astype(int)

    n_dup = len(adata.obs) - len(adata.obs[["array_row", "array_col"]].drop_duplicates())
    print(f"      网格: row_max={adata.obs['array_row'].max()} col_max={adata.obs['array_col'].max()} "
          f"重复格数={n_dup}", flush=True)
    return adata


# ---------------------------------------------------------------------------
# 2) SpaSEG 聚类
# ---------------------------------------------------------------------------
def run_spaseg(adata):
    from spaseg import spaseg  # noqa: 延迟 import（源码模块级会打印 scanpy 头）

    print_header_step(3, 6, "SpaSEG 卷积分割训练（CPU，最简参数）")
    t0 = time.time()

    spaseg_model = spaseg.SpaSEG(
        adata=[adata],
        use_gpu=False,
        device="cpu",
        seed=1029,
        input_dim=COMPONS,
        nChannel=COMPONS,
        output_dim=COMPONS,
        nConv=2,
        lr=0.002,
        weight_decay=1e-5,
        pretrain_epochs=PRETRAIN_EPOCHS,
        iterations=ITERATIONS,
        sim_weight=ALPHA,
        con_weight=BETA,
        min_label=MIN_LABEL,
        spot_size=None,
    )
    # 构造 image-like 三维输入 (n_batch, input_dim, H, W)
    input_mxt, H, W = spaseg_model._prepare_data()
    print(f"      input_mxt 形状 = {input_mxt.shape}, H={H}, W={W}", flush=True)

    cluster_label, embedding = spaseg_model._train(input_mxt)

    # 把分割标签回填到每个 spot，生成 obs['SpaSEG_clusters']
    spaseg_model._add_seg_label(cluster_label, 1, H, W, barcode_index="index")

    n_domains = adata.obs["SpaSEG_clusters"].nunique()
    print(f"      空间域数量 = {n_domains}: {sorted(adata.obs['SpaSEG_clusters'].unique().astype(str).tolist())}", flush=True)
    print(f"      聚类耗时 = {time.time() - t0:.1f}s", flush=True)
    return adata


# ---------------------------------------------------------------------------
# 3) SVG 检测
# ---------------------------------------------------------------------------
def merge_small_domains(adata, min_size=2):
    """把样本数 < min_size 的孤立小域，按空间最近邻并入邻近大域。

    scanpy 的 wilcoxon 差异检验要求每组至少 2 个样本；稀疏 STARmap 数据中
    SpaSEG 可能分出仅含 1 个 spot 的域，这里做一次类似 SpaGCN refine 的
    邻域合并，保证后续 detect_svg 可运行。
    """
    labels = adata.obs["SpaSEG_clusters"].astype(str)
    counts = labels.value_counts()
    small = counts[counts < min_size].index.tolist()
    if not small:
        return adata

    print(f"      有 {len(small)} 个过小空间域 {small}，按空间最近邻并入邻近大域", flush=True)
    from sklearn.neighbors import NearestNeighbors

    xy = adata.obsm["spatial"]
    nn = NearestNeighbors(n_neighbors=min(20, len(adata) - 1)).fit(xy)
    _, idxs = nn.kneighbors(xy)
    lab = labels.values
    big_label = counts.index[0]  # 最大域标签（兜底）

    new_lab = lab.copy()
    for i, l in enumerate(lab):
        if l in small:
            assigned = False
            for j in idxs[i]:  # 按距离升序找最近的非小域 spot
                if lab[j] not in small:
                    new_lab[i] = lab[j]
                    assigned = True
                    break
            if not assigned:
                new_lab[i] = big_label

    # 重新编码为 0..k-1 连续整型 category
    codes, _ = pd.factorize(new_lab)
    adata.obs["SpaSEG_clusters"] = pd.Categorical(codes)
    n_domains = adata.obs["SpaSEG_clusters"].nunique()
    print(f"      合并后空间域数量 = {n_domains}", flush=True)
    return adata


def detect_svgs(adata):
    from downstream.svg import detect_svg  # noqa

    print_header_step(4, 6, "逐空间域检测 SVG（detect_svg, 官方默认过滤）")
    # 处理仅含 1 个 spot 的域（wilcoxon 需要每组 >=2）
    adata = merge_small_domains(adata, min_size=2)
    # STARmap var 无 'mt' 列（只有 'mito'），故 filter_mt=False；
    # use_log=False 时 _data_prep 会把 log1p 表达 expm1 回伪 counts 再做 Wilcoxon（官方默认）。
    svg_df, _adata = detect_svg(
        adata,
        target_domains="all",
        filter_mt=False,
        use_log=False,
        domain_labels="SpaSEG_clusters",
        do_filter=True,
    )
    if svg_df is None or len(svg_df) == 0:
        print("      严格过滤后 0 条，降级为不过滤输出全量排名...", flush=True)
        svg_df, _adata = detect_svg(
            adata,
            target_domains="all",
            filter_mt=False,
            use_log=False,
            domain_labels="SpaSEG_clusters",
            do_filter=False,
        )
    print(f"      共检测到 SVG 记录 = {len(svg_df)}", flush=True)
    return svg_df, _adata


# ---------------------------------------------------------------------------
# 4) 结果保存 + 绘图
# ---------------------------------------------------------------------------
def plot_domains(adata, outdir):
    """空间域展示图。"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    xy = adata.obsm["spatial"]
    colors = adata.obs["SpaSEG_clusters"].astype("category")
    scatter = ax.scatter(xy[:, 0], xy[:, 1], c=colors.cat.codes, cmap="tab20",
                         s=12, linewidths=0)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("SpaSEG spatial domains")
    fig.colorbar(scatter, ax=ax, ticks=range(len(colors.cat.categories)), label="domain")
    out = outdir / f"domains_spaSEG_{SAMPLE_NAME}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      已保存空间域图: {out}", flush=True)
    return out


def plot_top_svgs(adata, svg_df, outdir):
    """Top SVG 基因的空间表达图。"""
    genes = svg_df["gene"].tolist()[:TOP_N]
    print(f"      Top SVG 基因: {genes}", flush=True)
    xy = adata.obsm["spatial"]

    # 用 X 的 log1p 表达（含 0）着色
    if hasattr(adata.X, "toarray"):
        expr_mat = adata.X.toarray()
    else:
        expr_mat = np.asarray(adata.X)

    for g in genes:
        if g not in adata.var_names:
            continue
        idx = list(adata.var_names).index(g)
        expr = expr_mat[:, idx]
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        scm = ax.scatter(xy[:, 0], xy[:, 1], c=expr, cmap="viridis", s=12, linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(g)
        fig.colorbar(scm, ax=ax, label="log1p expr")
        out = outdir / f"SVG_spaSEG_{SAMPLE_NAME}_{g}.png"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"      已保存 Top SVG 空间表达图到 {outdir}", flush=True)


def main():
    t_start = time.time()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # 1) 读取
    adata = load_data()
    # 2) 预处理
    adata = preprocess(adata)
    # 3) SpaSEG 聚类
    adata = run_spaseg(adata)
    # 4) SVG 检测
    svg_df, _adata = detect_svgs(adata)

    # 5) 保存 CSV
    print_header_step(5, 6, "保存 SVG 结果")
    csv_path = OUTDIR / f"SVG_spaSEG_{SAMPLE_NAME}.csv"
    svg_df.to_csv(csv_path, index=False)
    print(f"      已保存 SVG 结果: {csv_path} ({len(svg_df)} 条记录)", flush=True)

    # 6) 绘图
    print_header_step(6, 6, "绘制空间展示图")
    plot_domains(adata, OUTDIR)
    plot_top_svgs(adata, svg_df, OUTDIR)

    print(f"===== run_spaSEG 完成, 总耗时 {time.time() - t_start:.1f}s =====", flush=True)


if __name__ == "__main__":
    main()
