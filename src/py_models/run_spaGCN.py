# -*- coding: utf-8 -*-
"""
run_spaGCN.py —— 在 mouse_brain_STARmap 数据上运行 SpaGCN 检测 SVG

输入：data/STARmap/mouse_brain_cortex/mouse_brain_STARmap_processed.h5ad
      （坐标位于 obsm["spatial"]，无组织学图像）
输出目录：results/local_results/mouse_brain_STARmap/spaGCN/
      - SVG_spaGCN_STARmap_Mouse_Brain.csv    : 每个空间域检测出的 SVG 及统计指标
      - SVG_spaGCN_*.png                     : Top SVG 空间表达展示图
      - domains_spaGCN_STARmap_Mouse_Brain.png: SpaGCN 空间域划分图

流程（均为 SpaGCN 默认/最简参数）：
  1. 预处理：prefilter_genes(min_cells=3) + normalize + log1p
  2. 基于空间坐标（histology=False）计算邻接矩阵
  3. search_l(p=0.5) 找超参数 l
  4. search_res(target_num=n_clusters) 找聚类分辨率 res
  5. SpaGCN 训练 + 预测空间域，并用 refine() 平滑（square 形状，适配非 Visium）
  6. 对每个空间域用 detect_SVGs_ez_mode() 检测 SVG
  7. 汇总为 CSV 并绘制 Top SVG 空间表达图

用法（在项目根目录 F:\\computatinalbiology 下）:
    F:/computatinalbiology/env_spatial/python.exe ./src/r_models/run_spaGCN.py
"""
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

# 项目根目录（本文件位于 src/r_models/ 下，parents[2] = 项目根）
ROOT = Path(__file__).resolve().parents[2]

H5AD_PATH = ROOT / "data" / "STARmap" / "mouse_brain_cortex" / "mouse_brain_STARmap_processed.h5ad"
OUTDIR = ROOT / "results" / "local_results" / "mouse_brain_STARmap" / "spaGCN"
SAMPLE_NAME = "STARmap_Mouse_Brain"

# SpaGCN 默认/最简参数（来自 ez_mode）
N_CLUSTERS = 9          # 空间域数量（与数据自带的 clusters 类别数一致）
P = 0.5                 # search_l 的 p 参数（默认）
HISTOLOGY = False       # STARmap 无组织学图像，仅用空间坐标建图
COLOR_MAP = "viridis"   # SVG 展示图配色


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] 读取 h5ad: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH)
    print(f"      shape = {adata.shape} (spots x genes)")

    # 坐标：STARmap 的坐标在 obsm["spatial"]，列顺序为 [x, y]
    spatial = np.asarray(adata.obsm["spatial"], dtype=float)
    x_array = spatial[:, 0].tolist()
    y_array = spatial[:, 1].tolist()
    print(f"      坐标来源: obsm['spatial'], 形状 {spatial.shape}")

    # ---------- 1) 预处理（与官方教程一致的默认流程） ----------
    print("[2/6] 预处理（prefilter_genes + normalize + log1p）")
    adata.var_names_make_unique()
    SpaGCN_util_prefilter_genes(adata, min_cells=3)   # 去除在 <3 个 spot 表达的基因
    SpaGCN_util_prefilter_specialgenes(adata)         # 去除 ERCC / MT- 开头基因
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    print(f"      预处理后 shape = {adata.shape}")

    # ---------- 2) 计算邻接矩阵（无组织学图像） ----------
    print("[3/6] 计算邻接矩阵（histology=False，仅用空间坐标）")
    from SpaGCN.calculate_adj import calculate_adj_matrix
    adj = calculate_adj_matrix(x=x_array, y=y_array, histology=HISTOLOGY)
    print(f"      adj 形状 = {adj.shape}")

    # ---------- 3) 搜索超参数 l ----------
    print("[4/6] 搜索 l 与 res")
    from SpaGCN.util import search_l, search_res
    l = search_l(p=P, adj=adj)
    print(f"      l = {l}")

    # ---------- 4) 搜索聚类分辨率 res 并训练 ----------
    res = search_res(adata, adj, l, target_num=N_CLUSTERS)
    print(f"      res = {res}")

    print("[5/6] SpaGCN 训练 + 预测空间域")
    from SpaGCN import SpaGCN
    clf = SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="louvain", res=res, tol=5e-3, lr=0.05, max_epochs=200)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = y_pred.astype(str)

    # 平滑（STARmap 非六边形排布，用 square 形状）
    from SpaGCN.util import refine
    from SpaGCN.calculate_adj import calculate_adj_matrix as _calc_adj2d
    adj_2d = _calc_adj2d(x=x_array, y=y_array, histology=False)
    refined = refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(),
                     dis=adj_2d, shape="square")
    adata.obs["refined_pred"] = refined
    adata.obs["refined_pred"] = adata.obs["refined_pred"].astype(str)

    domains = sorted(adata.obs["refined_pred"].unique())
    print(f"      空间域数量 = {len(domains)}: {domains}")

    # ---------- 5) 检测每个域的 SVG ----------
    print("[6/6] 逐空间域检测 SVG")
    from SpaGCN.ez_mode import detect_SVGs_ez_mode, plot_spatial_domains_ez_mode

    x_name = "x"
    y_name = "y"
    adata.obs[x_name] = x_array
    adata.obs[y_name] = y_array

    # 域划分展示图
    plot_spatial_domains_ez_mode(
        adata, domain_name="refined_pred", x_name=x_name, y_name=y_name,
        plot_color=sc.pl.palettes.default_102, size=40,
        save_dir=str(OUTDIR / f"domains_spaGCN_{SAMPLE_NAME}.png"))

    # 对每个域检测 SVG（默认阈值）
    all_svg = []
    for dom in domains:
        try:
            df = detect_SVGs_ez_mode(
                adata, target=dom, x_name=x_name, y_name=y_name,
                domain_name="refined_pred",
                min_in_group_fraction=0.7,
                min_in_out_group_ratio=1.0,
                min_fold_change=1.5)
            df["domain"] = dom
            all_svg.append(df)
        except Exception as e:
            print(f"      [警告] 空间域 {dom} 检测 SVG 失败: {e}")

    if all_svg:
        svg_df = pd.concat(all_svg, ignore_index=True)
    else:
        svg_df = pd.DataFrame(
            columns=["genes", "in_group_fraction", "out_group_fraction",
                     "in_out_group_ratio", "in_group_mean_exp", "out_group_mean_exp",
                     "fold_change", "pvals_adj", "target_dmain", "neighbors", "domain"])

    csv_path = OUTDIR / f"SVG_spaGCN_{SAMPLE_NAME}.csv"
    svg_df.to_csv(csv_path, index=False)
    print(f"      已保存 SVG 结果: {csv_path} ({len(svg_df)} 条记录)")

    # ---------- 6) 绘制 Top SVG 空间表达图 ----------
    if len(svg_df) > 0:
        top_genes = svg_df.sort_values("pvals_adj")["genes"].head(6).tolist()
        print(f"      Top SVG 基因: {top_genes}")
        # 注：官方 plot_SVGs_ez_mode 在新版 scipy/pandas 下有稀疏矩阵赋值 bug，
        # 这里自定义绘图逻辑，等价实现（每个基因一张空间散点图）。
        plot_top_svgs(adata, top_genes, x_name, y_name, COLOR_MAP, OUTDIR, SAMPLE_NAME)
        print(f"      已保存 Top SVG 空间表达图到 {OUTDIR}")

    print("===== run_spaGCN 完成 =====")


def plot_top_svgs(adata, gene_list, x_name, y_name, color_map, outdir, sample_name):
    """绘制 Top SVG 基因的空间表达图（绕开官方 ez_mode 的稀疏矩阵 bug）。"""
    import matplotlib.pyplot as plt

    for g in gene_list:
        # 稀疏列转 dense 一维数组（等价官方 adata.obs["exp"] 的意图）
        col = adata.X[:, adata.var.index == g]
        if hasattr(col, "toarray"):
            expr = np.asarray(col.toarray()).ravel()
        else:
            expr = np.asarray(col).ravel()
        tmp = adata.copy()
        tmp.obs["exp"] = expr
        fig = sc.pl.scatter(tmp, alpha=1, x=x_name, y=y_name, color="exp",
                            title=g, color_map=color_map, show=False, size=40)
        fig.set_aspect("equal", "box")
        fig.axes.invert_yaxis()
        fig.figure.savefig(str(outdir / f"SVG_spaGCN_{sample_name}_{g}.png"), dpi=600)
        plt.close(fig.figure)


# 延迟导入 SpaGCN 工具函数（避免未使用时加载 torch）
def SpaGCN_util_prefilter_genes(adata, min_cells=3):
    from SpaGCN.util import prefilter_genes
    prefilter_genes(adata, min_cells=min_cells)


def SpaGCN_util_prefilter_specialgenes(adata):
    from SpaGCN.util import prefilter_specialgenes
    prefilter_specialgenes(adata)


if __name__ == "__main__":
    main()
