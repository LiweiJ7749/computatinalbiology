# -*- coding: utf-8 -*-
"""run_spaSEG.py —— SpaSEG 方法：SVG 检测（工程化/批量化版本）
================================================================================
设计（配合 src/__init__.py 的初始化模块）：
  1. 先解析 run 配置（src.resolve_run）：--dataset / --h5ad / --spatial /
     --outdir / --sample；命中数据集注册表即可零参数运行。
  2. 输入优先取共同前处理生成的 <sample>_spaSEG.h5ad（X + obsm['spatial']）；
     若不存在（未先跑 init）则回退读原始 h5ad 并自动补坐标。
  3. 预处理：表达已是 log1p+normalize，直接 PCA 作为 SpaSEG 输入特征；
     连续坐标等比缩放到 POSITION_MAX 上限后取整为紧凑整数网格
     (obs['array_row'] / obs['array_col'])，与官方对 cell-level 平台一致。
  4. SpaSEG 卷积分割训练 -> SpaSEG_clusters 空间域 -> detect_svg()（Wilcoxon）
     -> 过滤 -> SVG 结果。
  5. 输出 SVG_spaSEG_<sample>.csv + 空间域图 + Top SVG 空间表达图。

深度学习设备：默认 auto（有 CUDA 用 GPU，否则 CPU+警告）；可用 --device 强制
cuda/cpu。SpaSEG 使用本地源码 src/vendor/SpaSEG_src（勿用 site-packages 版）。

用法（项目根，用 envs/spatial 的 python，先跑 h5ad_preprocess.py 生成输入更佳）：
    python src/py_models/run_spaSEG.py --dataset mouse_brain_STARmap
    python src/py_models/run_spaSEG.py --h5ad ./data/.../x.h5ad \\
        --outdir ./results/local_results/my_run --sample my --device cuda
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 项目路径 & 引入 src 初始化模块（import 无副作用、开销低）
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402

# 让 SpaSEG 源码可被 import（本地 GPU 版源码；Windows/Linux 共用，SVG_EXT_DIR 可覆盖）
SPASEG_SRC = src.external_src_dir("SpaSEG_src")
if str(SPASEG_SRC) not in sys.path:
    sys.path.insert(0, str(SPASEG_SRC))

DEVICE_DEFAULT = "auto"     # auto = 有 GPU 用 cuda，否则 CPU+警告

# --- 从 configs/model_params/spaSEG.json 加载参数 ---
_PARAMS = src.load_model_params("spaseg")

# 兼容旧 CLI 默认值（当 config 键缺失时使用）
_DEFAULTS = {
    "pca_dim": 15,
    "alpha": 0.4,
    "beta": 0.7,
    "min_label": 7,
    "pretrain_epochs": 400,
    "iterations": 2100,
    "position_max": 500.0,
    "nConv": 2,
    "lr": 0.002,
    "weight_decay": 1e-5,
    "seed": 1029,
}
TOP_N = 6  # 绘制 Top N 个 SVG 的空间表达图
# 超过该 spot 数时，detect_svg 内部 _data_prep/_ranks_svg 会把整矩阵 toarray()/to_df()
# （50 万 x 1.9 万 ≈ 77GB float64，多份副本）导致 OOM，改用稀疏感知的轻量 SVG 检测。
LARGE_N_SPOTS = 100_000


def _get_param(key, default=None):
    """从 config 获取参数，config 缺失则用 _DEFAULTS。"""
    return _PARAMS.get(key, _DEFAULTS.get(key, default))


def _resolve_inputs(args):
    """解析 run 配置，返回 (adata路径, outdir, sample, 参数字典)。"""
    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad,
                          spatial=args.spatial, outdir=args.outdir,
                          sample=args.sample, methods=["spaseg"])
    src.ensure_run_dirs(run)
    outdir = run["method_dirs"]["spaseg"]
    sample = run["sample"]
    # 优先：共同前处理生成的 <sample>_spaSEG.h5ad（X + obsm['spatial']）
    prepared = outdir / f"{sample}_spaSEG.h5ad"
    h5ad_in = prepared if prepared.exists() else run["h5ad"]
    if prepared.exists():
        src.log_message(f"使用前处理产物: {prepared}")
    else:
        src.log_message(f"未发现 {prepared}，回退读原始 h5ad: {run['h5ad']}")
    params = dict(
        pca_dim=args.pca_dim or _get_param("pca_dim"),
        alpha=args.alpha if args.alpha is not None else _get_param("alpha"),
        beta=args.beta if args.beta is not None else _get_param("beta"),
        min_label=args.min_label or _get_param("min_label"),
        pretrain_epochs=args.pretrain_epochs or _get_param("pretrain_epochs"),
        iterations=args.iterations or _get_param("iterations"),
        position_max=args.position_max or _get_param("position_max"),
        nConv=_get_param("nConv"),
        lr=_get_param("lr"),
        weight_decay=_get_param("weight_decay"),
        seed=_get_param("seed"),
    )
    return h5ad_in, outdir, sample, params


def _pick_device(arg: str) -> str:
    """选择训练设备：auto -> cuda 优先；显式 cuda 但无 GPU 则报错。"""
    import torch

    has_cuda = torch.cuda.is_available()
    if arg == "auto":
        dev = "cuda" if has_cuda else "cpu"
        if not has_cuda:
            src.log_message("警告：未检测到 CUDA，回退 CPU"
                  "（默认 2100 迭代会很慢，可用 --iterations 调低）")
    else:
        dev = arg
    if dev == "cuda" and not has_cuda:
        raise SystemExit("[run_spaSEG] --device cuda 但未检测到可用 CUDA GPU，"
                         "请检查 nvidia-smi 与 torch 是否为 CUDA 版，或改用 --device auto/cpu。")
    if dev == "cuda":
        src.log_message(f"使用设备: cuda ({torch.cuda.get_device_name(0)})")
    return dev


# ---------------------------------------------------------------------------
# 1) 读取 + 预处理
# ---------------------------------------------------------------------------
def load_data(h5ad_in: Path):
    import scanpy as sc

    src.log_step(1, 6, f"读取 h5ad: {h5ad_in}")
    adata = sc.read_h5ad(h5ad_in)
    src.log_message(f"shape = {adata.shape} (spots x genes)")
    src.log_message(f"X 类型 = {type(adata.X).__name__}, max = {adata.X.max():.3f}")
    if "spatial" not in adata.obsm:
        raise ValueError("h5ad 缺少 obsm['spatial'] 坐标（先跑 src/preprocess/h5ad_preprocess.py）")
    src.log_message(f"坐标来源 = obsm['spatial'], 形状 = {adata.obsm['spatial'].shape}")
    return adata


def preprocess(adata, params):
    """表达 PCA + 构造 SpaSEG 所需的整数网格坐标 array_row/array_col。"""
    src.log_step(2, 6, "预处理（PCA 特征 + 坐标网格化）")

    # 2.1 表达特征：X 已是 log1p+normalize，直接 PCA 作为 SpaSEG 的输入表达
    compons = params["pca_dim"]
    if "X_pca" not in adata.obsm:
        import scanpy as sc

        sc.pp.pca(adata, n_comps=compons, random_state=0)
    else:
        adata.obsm["X_pca"] = adata.obsm["X_pca"][:, :compons]
    src.log_message(f"X_pca 形状 = {adata.obsm['X_pca'].shape}")

    # 2.2 坐标网格化：连续坐标平移>=0 后等比缩放（最大边 -> POSITION_MAX），再取整
    position_max = params["position_max"]
    xy = adata.obsm["spatial"].astype(np.float64)
    scale = position_max / np.max(xy)
    arr_row = (xy[:, 0] - xy[:, 0].min()) * scale
    arr_col = (xy[:, 1] - xy[:, 1].min()) * scale

    adata.obs["array_row"] = arr_row.astype(int)
    adata.obs["array_col"] = arr_col.astype(int)

    n_dup = len(adata.obs) - len(adata.obs[["array_row", "array_col"]].drop_duplicates())
    src.log_message(f"网格: row_max={adata.obs['array_row'].max()} "
          f"col_max={adata.obs['array_col'].max()} 重复格数={n_dup}")
    return adata


# ---------------------------------------------------------------------------
# 2) SpaSEG 聚类
# ---------------------------------------------------------------------------
def run_spaseg(adata, device: str, params):
    from spaseg import spaseg  # noqa: 延迟 import（源码模块级会打印 scanpy 头）

    src.log_step(3, 6, f"SpaSEG 卷积分割训练（device={device}）")
    use_gpu = device == "cuda"
    t0 = time.time()

    spaseg_model = spaseg.SpaSEG(
        adata=[adata],
        use_gpu=use_gpu,
        device=device,
        seed=params.get("seed", 1029),
        input_dim=params["pca_dim"],
        nChannel=params["pca_dim"],
        output_dim=params["pca_dim"],
        nConv=params.get("nConv", 2),
        lr=params.get("lr", 0.002),
        weight_decay=params.get("weight_decay", 1e-5),
        pretrain_epochs=params["pretrain_epochs"],
        iterations=params["iterations"],
        sim_weight=params["alpha"],
        con_weight=params["beta"],
        min_label=params["min_label"],
        spot_size=None,
    )
    # 构造 image-like 三维输入 (n_batch, input_dim, H, W)
    input_mxt, H, W = spaseg_model._prepare_data()
    src.log_message(f"input_mxt 形状 = {input_mxt.shape}, H={H}, W={W}")

    cluster_label, embedding = spaseg_model._train(input_mxt)

    # 把分割标签回填到每个 spot，生成 obs['SpaSEG_clusters']
    spaseg_model._add_seg_label(cluster_label, 1, H, W, barcode_index="index")

    n_domains = adata.obs["SpaSEG_clusters"].nunique()
    src.log_message(f"空间域数量 = {n_domains}: "
          f"{sorted(adata.obs['SpaSEG_clusters'].unique().astype(str).tolist())}")
    src.log_message(f"聚类耗时 = {time.time() - t0:.1f}s")
    return adata


# ---------------------------------------------------------------------------
# 3) SVG 检测
# ---------------------------------------------------------------------------
def merge_small_domains(adata, min_size=2):
    """把样本数 < min_size 的孤立小域，按空间最近邻并入邻近大域。"""
    labels = adata.obs["SpaSEG_clusters"].astype(str)
    counts = labels.value_counts()
    small = counts[counts < min_size].index.tolist()
    if not small:
        return adata

    src.log_message(f"有 {len(small)} 个过小空间域 {small}，按空间最近邻并入邻近大域")
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
    src.log_message(f"合并后空间域数量 = {n_domains}")
    return adata


def _detect_svgs_sparse(adata):
    """大 spot 数据的轻量 SVG 检测（稀疏感知，替代 detect_svg 的密集计算）。

    detect_svg 内部 ``_data_prep`` 的 ``adata_.X.toarray()`` 与 ``_ranks_svg`` 的
    ``adata.to_df()`` 会把整矩阵 (n_spots x n_genes) 转 dense，50 万级 spot 必然 OOM。
    这里只用 ``sc.tl.rank_genes_groups``（逐基因、稀疏切片）做域差异检验，再按
    padj < 0.05 提取每域显著基因，避免触碰整矩阵 dense 化。
    """
    import scanpy as sc

    src.log_message(f"大 spot 数据（{adata.n_obs} spots）：detect_svg 整矩阵 dense 化会 OOM，"
                    "改用轻量 rank_genes_groups")
    adata.obs["SpaSEG_clusters"] = adata.obs["SpaSEG_clusters"].astype(str).astype("category")
    sc.tl.rank_genes_groups(
        adata, groupby="SpaSEG_clusters", reference="rest",
        n_genes=adata.shape[1], method="wilcoxon")

    rgg = adata.uns["rank_genes_groups"]
    domains = list(rgg["names"].dtype.names)
    rows = []
    for d in domains:
        names = rgg["names"][d].astype(str)
        padjs = np.asarray(rgg["pvals_adj"][d], dtype=float)
        lfcs = np.asarray(rgg["logfoldchanges"][d], dtype=float)
        for g, p, fc in zip(names, padjs, lfcs):
            if np.isfinite(p):
                rows.append({"gene": g, "domain": d,
                             "pvals_adj": float(p), "log2_fc": float(fc)})
    svg_df = pd.DataFrame(rows)
    if len(svg_df):
        svg_df = svg_df[svg_df["pvals_adj"] < 0.05].reset_index(drop=True)
    src.log_message(f"轻量 SVG 检测完成，共 {len(svg_df)} 条")
    return svg_df, adata


def detect_svgs(adata):
    # 处理仅含 1 个 spot 的域（wilcoxon 需要每组 >=2）
    adata = merge_small_domains(adata, min_size=2)
    # 单域无法做“域内 vs 域外”比较，vendor _ranks_svg 会因 out_groups 为空而崩溃
    if adata.obs["SpaSEG_clusters"].nunique() <= 1:
        src.log_message("空间域数量 <= 1，无法做域间差异 SVG 检测，返回空结果")
        return pd.DataFrame(), adata
    if adata.n_obs > LARGE_N_SPOTS:
        return _detect_svgs_sparse(adata)

    from downstream.svg import detect_svg  # noqa

    # 兼容 dense X（如 drosophila Stereo-seq）：vendor detect_svg 的 _data_prep
    # 假设 X 是稀疏矩阵并调用 X.toarray()，dense 时需先转 csr。
    from scipy import sparse
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)

    src.log_step(4, 6, "逐空间域检测 SVG（detect_svg, 官方默认过滤）")
    # STARmap var 无 'mt' 列（只有 'mito'），故 filter_mt=False；
    # use_log=False 时 _data_prep 会把 log1p 表达 expm1 回伪 counts 再做 Wilcoxon（官方默认）。
    try:
        svg_df, _adata = detect_svg(
            adata,
            target_domains="all",
            filter_mt=False,
            use_log=False,
            domain_labels="SpaSEG_clusters",
            do_filter=True,
        )
        if svg_df is None or len(svg_df) == 0:
            src.log_message("严格过滤后 0 条，降级为不过滤输出全量排名...")
            svg_df, _adata = detect_svg(
                adata,
                target_domains="all",
                filter_mt=False,
                use_log=False,
                domain_labels="SpaSEG_clusters",
                do_filter=False,
            )
    except Exception as e:
        src.log_message(f"detect_svg 失败: {e}，返回空结果")
        return pd.DataFrame(), adata
    src.log_message(f"共检测到 SVG 记录 = {len(svg_df)}")
    return svg_df, _adata


# ---------------------------------------------------------------------------
# 4) 结果保存 + 绘图
# ---------------------------------------------------------------------------
def save_gene_ranking(_adata, outdir, sample, domain_labels="SpaSEG_clusters"):
    """从 rank_genes_groups 结果提取全基因排名（列固定: gene, stat, pval, padj, rank）。"""
    rgg = _adata.uns.get("rank_genes_groups")
    if rgg is None:
        src.log_message("未找到 rank_genes_groups，跳过全基因排名")
        return
    domains = [d for d in rgg["names"].dtype.names]
    gene_min = {}
    gene_min_pval = {}
    has_pvals = "pvals" in rgg and rgg["pvals"] is not None
    for d in domains:
        genes = rgg["names"][d].astype(str)
        padjs = np.asarray(rgg["pvals_adj"][d], dtype=float)
        pvals = (np.asarray(rgg["pvals"][d], dtype=float)
                 if has_pvals else padjs)
        for g, p, pv in zip(genes, padjs, pvals):
            if not np.isfinite(p):
                continue
            if g not in gene_min or p < gene_min[g]:
                gene_min[g] = p
            if np.isfinite(pv) and (g not in gene_min_pval or pv < gene_min_pval[g]):
                gene_min_pval[g] = pv
    if not gene_min:
        src.log_message("全基因排名为空，跳过")
        return
    rank_df = pd.DataFrame({"gene": list(gene_min.keys()),
                            "padj": list(gene_min.values())})
    rank_df["stat"] = -np.log10(rank_df["padj"].clip(lower=1e-300))
    rank_df["pval"] = rank_df["gene"].map(gene_min_pval)
    rank_df = rank_df.sort_values(["padj", "gene"]).reset_index(drop=True)
    rank_df["rank"] = rank_df.index + 1
    rank_df = rank_df[["gene", "stat", "pval", "padj", "rank"]]
    rank_path = outdir / f"SVG_spaSEG_{sample}_rank.csv"
    rank_df.to_csv(rank_path, index=False)
    src.log_message(f"已保存全基因排名: {rank_path} ({len(rank_df)} 个基因)")


def plot_domains(adata, outdir, sample):
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
    out = outdir / f"domains_spaSEG_{sample}.png"
    fig.savefig(str(out), dpi=300, bbox_inches="tight")
    plt.close(fig)
    src.log_message(f"已保存空间域图: {out}")
    return out


def plot_top_svgs(adata, svg_df, outdir, sample):
    """Top SVG 基因的空间表达图。"""
    if svg_df is None or len(svg_df) == 0 or "gene" not in svg_df.columns:
        src.log_message("无 SVG 记录，跳过 Top 绘图")
        return
    genes = svg_df["gene"].tolist()[:TOP_N]
    src.log_message(f"Top SVG 基因: {genes}")
    xy = adata.obsm["spatial"]

    for g in genes:
        if g not in adata.var_names:
            continue
        idx = list(adata.var_names).index(g)
        # 逐基因取稀疏列，避免大 spot 数据整矩阵 toarray()（50 万 x 1.9 万 ≈ 77GB）OOM
        col = adata.X[:, idx]
        if hasattr(col, "toarray"):
            expr = np.asarray(col.toarray()).ravel()
        else:
            expr = np.asarray(col).ravel()
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        scm = ax.scatter(xy[:, 0], xy[:, 1], c=expr, cmap="viridis", s=12, linewidths=0)
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_title(g)
        fig.colorbar(scm, ax=ax, label="log1p expr")
        out = outdir / f"SVG_spaSEG_{sample}_{g}.png"
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        plt.close(fig)
    src.log_message(f"已保存 Top SVG 空间表达图到 {outdir}")


def main():
    ap = argparse.ArgumentParser(description="SpaSEG SVG 检测 (batch)")
    ap.add_argument("--dataset", default=None, help="dataset key")
    ap.add_argument("--h5ad", default=None, help="输入 h5ad（绝对或相对项目根）")
    ap.add_argument("--spatial", default=None, help="可选坐标文件")
    ap.add_argument("--outdir", default=None, help="输出根目录")
    ap.add_argument("--sample", default=None, help="样本标签（默认 dataset）")
    ap.add_argument("--device", default=DEVICE_DEFAULT, choices=["auto", "cuda", "cpu"])
    ap.add_argument("--pca-dim", type=int, default=None, help=f"PCA 维数（默认 config 或 {_DEFAULTS['pca_dim']}）")
    ap.add_argument("--alpha", type=float, default=None, help=f"sim_weight（默认 config 或 {_DEFAULTS['alpha']}）")
    ap.add_argument("--beta", type=float, default=None, help=f"con_weight（默认 config 或 {_DEFAULTS['beta']}）")
    ap.add_argument("--min-label", type=int, default=None, help=f"min_label（默认 config 或 {_DEFAULTS['min_label']}）")
    ap.add_argument("--pretrain-epochs", type=int, default=None,
                    help=f"预训练轮数（默认 config 或 {_DEFAULTS['pretrain_epochs']}）")
    ap.add_argument("--iterations", type=int, default=None,
                    help=f"正式迭代数（默认 config 或 {_DEFAULTS['iterations']}）")
    ap.add_argument("--position-max", type=float, default=None,
                    help=f"坐标缩放上限（默认 config 或 {_DEFAULTS['position_max']}）")
    args = ap.parse_args()

    t_start = time.time()
    src.log_header(f"SpaSEG: {args.sample or args.dataset or 'unknown'}")
    h5ad_in, outdir, sample, params = _resolve_inputs(args)
    device = _pick_device(args.device)

    # 1) 读取
    adata = load_data(h5ad_in)
    # 2) 预处理
    adata = preprocess(adata, params)
    # 3) SpaSEG 聚类
    adata = run_spaseg(adata, device, params)
    # 4) SVG 检测
    svg_df, _adata = detect_svgs(adata)

    # 5) 保存 CSV
    src.log_step(5, 6, "保存 SVG 结果")
    csv_path = outdir / f"SVG_spaSEG_{sample}.csv"
    svg_df.to_csv(csv_path, index=False)
    src.log_message(f"已保存 SVG 结果: {csv_path} ({len(svg_df)} 条记录)")
    # 全基因排名（detect_svg 内部已算 rank_genes_groups，此处复用其结果）
    save_gene_ranking(_adata, outdir, sample)

    # 6) 绘图
    src.log_step(6, 6, "绘制空间展示图")
    plot_domains(adata, outdir, sample)
    plot_top_svgs(adata, svg_df, outdir, sample)

    # 保存运行时间（JSON，供 evaluation 汇总效率指标）
    import json

    rt_path = outdir / "runtime.json"
    rt_path.write_text(json.dumps(
        {"method": "spaseg", "sample": sample,
         "wall_seconds": round(time.time() - t_start, 2)}))
    src.log_message(f"已保存运行时间: {rt_path}")

    total = time.time() - t_start
    src.log_message(f"总耗时 {total:.1f}s", section=f"SpaSEG {sample} 完成")


if __name__ == "__main__":
    main()
