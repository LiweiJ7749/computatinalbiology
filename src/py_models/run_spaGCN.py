# -*- coding: utf-8 -*-
"""
run_spaGCN.py —— SpaGCN 方法：SVG 检测（工程化/批量化版本）
================================================================================
设计（配合 src/__init__.py 的初始化模块）：
  1. 先解析 run 配置（src.resolve_run）：--dataset / --h5ad / --outdir / --sample
  2. 输入优先取共同前处理生成的 <sample>_spaGCN.h5ad（X=原始 counts + obs x/y）；
     若不存在（未先跑 init）则回退读原始 h5ad 并自动补 X=raw_count 与坐标。
  3. 预处理（SpaGCN 默认/最简）：prefilter_genes + prefilter_specialgenes +
     normalize_per_cell + log1p（在真实 counts 上做**单次**标准化）。
  4. calculate_adj_matrix(histology=False) -> search_l -> search_res -> 训练(device)
     -> predict + refine 平滑 -> 逐域 detect_SVGs_ez_mode。
  5. 输出 SVG_spaGCN_<sample>.csv + 域划分图 + Top SVG 空间表达图。

深度学习设备：默认 auto（有 CUDA 用 GPU，否则 CPU+警告）；可用 --device 强制。
SpaGCN 使用本地 GPU 补丁源码 src/vendor/SpaGCN_src（勿用 site-packages CPU 版）。

用法（在项目根下，用 envs/spatial 的 python）：
    python src/py_models/run_spaGCN.py --dataset mouse_brain_STARmap
    python src/py_models/run_spaGCN.py --h5ad ./data/.../x.h5ad --outdir ./results/local_results/my_run --sample my
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# 项目路径 & 引入 src 初始化模块（import 无副作用、开销低）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402

# 深度学习设备：要求 CUDA GPU（torch 需为 cu 构建）；SpaGCN_src 提供 device 参数
DEVICE_DEFAULT = "auto"   # auto=有 GPU 用 cuda；否则 CPU 回退


def _load_params():
    """从 configs/model_params/spaGCN.json 加载参数，与 CLI 覆盖合并。"""
    defaults = src.load_model_params("spagcn")
    return defaults


PARAMS = _load_params()


def _resolve_inputs(args):
    """解析 run 配置，返回 (adata路径, outdir, sample, tech)。"""
    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad,
                          spatial=args.spatial, outdir=args.outdir,
                          sample=args.sample, methods=["spagcn"])
    src.ensure_run_dirs(run)
    outdir = run["method_dirs"]["spagcn"]
    sample = run["sample"]
    tech = run.get("tech") or ""
    # 优先：共同前处理生成的 <sample>_spaGCN.h5ad（X=counts + obs x/y）
    prepared = outdir / f"{sample}_spaGCN.h5ad"
    h5ad_in = prepared if prepared.exists() else run["h5ad"]
    if prepared.exists():
        src.log_message(f"使用前处理产物: {prepared}")
    else:
        src.log_message(f"未发现 {prepared}，回退读原始 h5ad: {run['h5ad']}")
    return h5ad_in, outdir, sample, tech


def _pick_device(arg: str) -> str:
    """选择训练设备：auto -> cuda 优先；显式 cuda 但无 GPU 则报错。"""
    import torch

    has_cuda = torch.cuda.is_available()
    if arg == "auto":
        dev = "cuda" if has_cuda else "cpu"
        if not has_cuda:
            src.log_message("警告：未检测到 CUDA，回退 CPU（会很慢）")
    else:
        dev = arg
    if dev == "cuda" and not has_cuda:
        raise SystemExit("[run_spaGCN] --device cuda 但未检测到可用 CUDA GPU，"
                         "请检查 nvidia-smi 与 torch 是否为 CUDA 版，或改用 --device auto/cpu。")
    if has_cuda:
        src.log_message(f"使用设备: cuda ({torch.cuda.get_device_name(0)})")
    return dev


# ---------------------------------------------------------------------------
# 读取 + 表达预处理（SpaGCN 默认流程）
# ---------------------------------------------------------------------------
def load_and_prepare(h5ad_in: Path, device: str):
    import anndata as ad
    import scanpy as sc

    src.log_step(1, 6, f"读取 h5ad: {h5ad_in}")
    adata = ad.read_h5ad(h5ad_in)
    src.log_message(f"shape = {adata.shape} (spots x genes)")

    # 坐标列 x/y（SpaGCN 用于建邻接矩阵 / refine / 绘图）
    if "x" not in adata.obs.columns or "y" not in adata.obs.columns:
        src.log_message("obs 无 x/y，从 obsm['spatial'] 派生")
        coords = np.asarray(adata.obsm["spatial"], dtype=float)
        adata.obs["x"] = coords[:, 0]
        adata.obs["y"] = coords[:, 1]
    x_array = adata.obs["x"].astype(float).tolist()
    y_array = adata.obs["y"].astype(float).tolist()

    # X 统一为真实 counts（若存在 raw_count 层），再做单次 normalize+log1p
    if "raw_count" in adata.layers and adata.layers["raw_count"] is not None:
        adata.X = adata.layers["raw_count"].copy()
        src.log_message("X <- layers['raw_count']（真实 counts）")

    src.log_step(2, 6, "预处理（prefilter + normalize + log1p）")
    min_cells = PARAMS.get("min_cells", 3)
    adata.var_names_make_unique()
    _prefilter_genes(adata, min_cells=min_cells)
    _prefilter_specialgenes(adata)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    src.log_message(f"预处理后 shape = {adata.shape}")
    return adata, x_array, y_array, device


def _prefilter_genes(adata, min_cells=3):
    from SpaGCN.util import prefilter_genes

    prefilter_genes(adata, min_cells=min_cells)


def _prefilter_specialgenes(adata):
    from SpaGCN.util import prefilter_specialgenes

    prefilter_specialgenes(adata)


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------
def _infer_n_clusters(adata):
    """无 --n-clusters 时的域数推断：优先 h5ad 自带类别列，否则回退默认。

    依次探测常见类别列；都没有则用 PARAMS 的默认值（configs/model_params/spaGCN.json
    的 n_clusters，缺省 9），并打印提示让用户用 --n-clusters 覆盖。
    """
    for col in ("clusters", "cell_type", "leiden", "louvain", "celltype", "domain"):
        if col in adata.obs.columns and adata.obs[col].notna().any():
            n = int(adata.obs[col].nunique())
            src.log_message(f"使用类别列 obs['{col}'] 的类别数 {n} 作为域数")
            return n
    default = int(PARAMS.get("n_clusters") or 9)
    src.log_message(f"未发现类别标注列，目标域数回退默认 {default}（可用 --n-clusters 覆盖）")
    return default


def train_domains(adata, x_array, y_array, device, n_clusters: int, tech: str = ""):
    import random
    import scanpy as sc
    import torch
    from SpaGCN import SpaGCN
    from SpaGCN.calculate_adj import calculate_adj_matrix
    from SpaGCN.util import search_l, search_res

    # 固定随机种子（louvain 初始化 + torch 训练均依赖全局 RNG），保证域划分可复现
    r_seed = PARAMS.get("r_seed", 100)
    t_seed = PARAMS.get("t_seed", 100)
    n_seed = PARAMS.get("n_seed", 100)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)

    src.log_step(3, 6, "计算邻接矩阵 (histology=False) + 搜索超参 l / res")
    adj = calculate_adj_matrix(x=x_array, y=y_array, histology=False)
    p_val = PARAMS.get("p", 0.5)
    l = search_l(p=p_val, adj=adj)
    src.log_message(f"l = {l}")

    # 目标域数：优先 --n-clusters；其次 h5ad 自带类别列；默认 9
    if n_clusters is None:
        n_clusters = _infer_n_clusters(adata)
    src.log_message(f"target_num(clusters) = {n_clusters}")
    res = search_res(adata, adj, l, target_num=n_clusters)
    src.log_message(f"res = {res}")

    src.log_step(4, 6, "SpaGCN 训练 + 预测 + refine")
    tol = PARAMS.get("tol", 5e-3)
    lr = PARAMS.get("lr", 0.05)
    max_epochs = PARAMS.get("max_epochs", 200)
    # 训练前再次固定种子（search_res 内部会消耗 RNG 状态）
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    clf = SpaGCN()
    clf.set_l(l)
    clf.train(adata, adj, init_spa=True, init="louvain", res=res,
              tol=tol, lr=lr, max_epochs=max_epochs, device=device)
    y_pred, prob = clf.predict()
    adata.obs["pred"] = np.asarray(y_pred).astype(str)

    from SpaGCN.calculate_adj import calculate_adj_matrix as calc_adj2d
    from SpaGCN.util import refine

    # refine 邻域形状：Visium/DLPFC 为六边形 spot 网格 -> hexagon(6)；其余(ST/STARmap/MERFISH...) -> square(4)
    shape = "hexagon" if tech in ("Visium", "Visium_HD", "DLPFC") else "square"
    adj_2d = calc_adj2d(x=x_array, y=y_array, histology=False)
    refined = refine(sample_id=adata.obs.index.tolist(), pred=adata.obs["pred"].tolist(),
                     dis=adj_2d, shape=shape)
    adata.obs["refined_pred"] = np.asarray(refined).astype(str)
    domains = sorted(adata.obs["refined_pred"].unique())
    src.log_message(f"refine shape = {shape} | 空间域数量 = {len(domains)}")
    return adata, domains


# ---------------------------------------------------------------------------
# SVG 检测 + 保存 + 绘图
# ---------------------------------------------------------------------------
def detect_and_save(adata, domains, x_name, y_name, outdir, sample):
    from SpaGCN.calculate_adj import calculate_adj_matrix
    from SpaGCN.util import (search_radius, find_neighbor_clusters,
                             rank_genes_groups)
    from SpaGCN.ez_mode import plot_spatial_domains_ez_mode
    import scanpy as sc

    src.log_step(5, 6, "逐空间域检测 SVG（并计算全基因排名）")
    plot_spatial_domains_ez_mode(
        adata, domain_name="refined_pred", x_name=x_name, y_name=y_name,
        plot_color=sc.pl.palettes.default_102, size=40,
        save_dir=str(outdir / f"domains_spaGCN_{sample}.png"))

    cell_id = adata.obs.index.tolist()
    x = adata.obs[x_name].tolist()
    y = adata.obs[y_name].tolist()
    pred = adata.obs["refined_pred"].tolist()

    adj_2d = calculate_adj_matrix(x=x, y=y, histology=False)
    start, end = np.quantile(adj_2d[adj_2d != 0], q=0.001), \
        np.quantile(adj_2d[adj_2d != 0], q=0.1)

    all_svg = []
    gene_min_padj = {}   # 基因 -> 各域最小 pvals_adj（用于全基因排名口径）
    gene_min_pval = {}   # 基因 -> 各域最小 pvals（未校正，供统一 pval 列）
    gene_effect = {}     # 基因 -> 取 min padj 那个域的 |log2(fold_change)|（无方向域间效应量）

    for dom in domains:
        try:
            r = search_radius(target_cluster=dom, cell_id=cell_id, x=x, y=y,
                              pred=pred, start=start, end=end,
                              num_min=10, num_max=14, max_run=100)
            if r is None:
                r = start
            nbr = find_neighbor_clusters(target_cluster=dom, cell_id=cell_id,
                                         x=x, y=y, pred=pred, radius=r,
                                         ratio=1 / 2)[0:3]
            de_info = rank_genes_groups(input_adata=adata, target_cluster=dom,
                                        nbr_list=nbr, label_col="refined_pred",
                                        adj_nbr=True, log=True)
            fcs = de_info["fold_change"] if "fold_change" in de_info.columns else de_info["pvals_adj"]
            # 全基因排名：记录每个基因在各域的最小 padj，并同步记录该域的无方向
            # 域间效应量 |log2(fold_change)|（供评价侧 rank_vs_effect 单调性检验）
            for g, p, fc in zip(de_info["genes"], de_info["pvals_adj"], fcs):
                if g not in gene_min_padj or p < gene_min_padj[g]:
                    gene_min_padj[g] = p
                    fc_safe = float(np.clip(fc, 1e-6, 1e6)) if np.isfinite(fc) else 1.0
                    gene_effect[g] = abs(np.log2(fc_safe))
            # 未校正 p 值：与 padj 同一基因对齐记录最小 pval
            pvals = de_info["pvals"] if "pvals" in de_info.columns else de_info["pvals_adj"]
            for g, p in zip(de_info["genes"], pvals):
                if g not in gene_min_pval or p < gene_min_pval[g]:
                    gene_min_pval[g] = p
            # 集合口径：与官方 detect_SVGs_ez_mode 相同的过滤规则
            filt = de_info[(de_info["pvals_adj"] < 0.05) &
                           (de_info["in_out_group_ratio"] > 1.0) &
                           (de_info["in_group_fraction"] > 0.7) &
                           (de_info["fold_change"] > 1.5)].copy()
            filt["domain"] = dom
            all_svg.append(filt)
        except Exception as e:
            src.log_message(f"空间域 {dom} 检测 SVG 失败: {e}")

    if all_svg:
        svg_df = pd.concat(all_svg, ignore_index=True)
    else:
        svg_df = pd.DataFrame()
    csv_path = outdir / f"SVG_spaGCN_{sample}.csv"
    svg_df.to_csv(csv_path, index=False)
    src.log_message(f"已保存 SVG 结果: {csv_path} ({len(svg_df)} 条记录)")

    # ---- 全基因排名 CSV（列固定: gene, stat, pval, padj, rank）----
    if gene_min_padj:
        rank_df = pd.DataFrame(
            {"gene": list(gene_min_padj.keys()),
             "padj": list(gene_min_padj.values())})
        rank_df["stat"] = -np.log10(rank_df["padj"].clip(lower=1e-300))
        rank_df["pval"] = rank_df["gene"].map(gene_min_pval)
        rank_df["effect"] = rank_df["gene"].map(gene_effect)
        rank_df = rank_df.sort_values(["padj", "gene"]).reset_index(drop=True)
        rank_df["rank"] = rank_df.index + 1
        rank_df = rank_df[["gene", "stat", "pval", "padj", "effect", "rank"]]
        rank_path = outdir / f"SVG_spaGCN_{sample}_rank.csv"
        rank_df.to_csv(rank_path, index=False)
        src.log_message(f"已保存全基因排名: {rank_path} ({len(rank_df)} 个基因)")
    return svg_df


def plot_top_svgs(adata, svg_df, x_name, y_name, outdir, sample, top_n=6):
    """Top SVG 基因空间表达图（绕开官方 ez_mode 稀疏矩阵 bug，等价实现）。"""
    import matplotlib.pyplot as plt
    import scanpy as sc

    if svg_df is None or len(svg_df) == 0:
        src.log_message("无 SVG 记录，跳过 Top 绘图")
        return
    gene_col = "genes" if "genes" in svg_df.columns else svg_df.columns[0]
    top = svg_df.sort_values("pvals_adj")[gene_col].head(top_n).tolist()
    src.log_message(f"Top SVG 基因: {top}")
    for g in top:
        col = adata.X[:, adata.var.index == g]
        expr = np.asarray(col.toarray()).ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()
        tmp = adata.copy()
        tmp.obs["exp"] = expr
        fig = sc.pl.scatter(tmp, alpha=1, x=x_name, y=y_name, color="exp",
                            title=g, color_map="viridis", show=False, size=40)
        fig.set_aspect("equal", "box")
        fig.axes.invert_yaxis()
        fig.figure.savefig(str(outdir / f"SVG_spaGCN_{sample}_{g}.png"), dpi=300)
        plt.close(fig.figure)
    src.log_message(f"已保存 Top SVG 空间表达图到 {outdir}")


def main():
    ap = argparse.ArgumentParser(description="SpaGCN SVG 检测 (batch)")
    ap.add_argument("--dataset", default=None, help="dataset key")
    ap.add_argument("--h5ad", default=None, help="输入 h5ad（绝对或相对项目根）")
    ap.add_argument("--spatial", default=None, help="可选坐标文件")
    ap.add_argument("--outdir", default=None, help="输出根目录")
    ap.add_argument("--sample", default=None, help="样本标签（默认 dataset）")
    ap.add_argument("--device", default=DEVICE_DEFAULT, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--n-clusters", type=int, default=None, help="目标空间域数")
    args = ap.parse_args()

    t_start = time.time()
    h5ad_in, outdir, sample, tech = _resolve_inputs(args)
    abs_outdir = outdir.resolve() if isinstance(outdir, Path) else Path(outdir).resolve()
    src.log_header(f"SpaGCN: {sample}")
    device = _pick_device(args.device)

    # 数据集级运行参数（configs/run_params.json）：CLI 未指定 --n-clusters 时用它覆盖
    run_params = src.load_run_params(args.dataset or "")
    if args.n_clusters is None:
        args.n_clusters = run_params.get("spagcn", {}).get("n_clusters")

    # 让 SpaGCN 源码可被 import（本地 GPU 补丁版；Windows/Linux 共用，SVG_EXT_DIR 可覆盖）
    spagcn_src = src.external_src_dir("SpaGCN_src")
    if str(spagcn_src) not in sys.path:
        sys.path.insert(0, str(spagcn_src))

    adata, x_array, y_array, device = load_and_prepare(h5ad_in, device)
    adata, domains = train_domains(adata, x_array, y_array, device, args.n_clusters, tech)

    # 坐标列作为绘图用 x_name/y_name
    x_name, y_name = "x", "y"
    svg_df = detect_and_save(adata, domains, x_name, y_name, outdir, sample)
    src.log_step(6, 6, "绘制 Top SVG 空间表达图")
    plot_top_svgs(adata, svg_df, x_name, y_name, outdir, sample)

    # 保存运行时间（JSON，供 evaluation 汇总效率指标）
    import json

    rt_path = outdir / "runtime.json"
    rt_path.write_text(json.dumps(
        {"method": "spagcn", "sample": sample,
         "wall_seconds": round(time.time() - t_start, 2)}))
    src.log_message(f"已保存运行时间: {rt_path}")

    total = time.time() - t_start
    src.log_message(f"总耗时 {total:.1f}s", section=f"SpaGCN {sample} 完成")


if __name__ == "__main__":
    main()
