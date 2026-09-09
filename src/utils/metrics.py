# -*- coding: utf-8 -*-
"""metrics.py —— 无真值条件下四方法 SVG 检测的核心评价计算

依据 `paper_material/方法评价指标.md` 的"统一评价协议"，在当前项目现状
（本地 2 个数据集、无模拟真值、无 marker 列表）下，落地**可执行**的指标：

  维度 1 正确性（第三方统计量，方法无关）：
    1.8  检出集合 Moran's I / Geary's C（k=6 近邻二元权重）+ 随机对照置换检验
  维度 3 统计性质：
    3.1  p 值直方图形态（全基因 pval）
    3.2  显著比例随 FDR 阈值增长曲线
    3.3  效应量单调性（方法排名 vs 独立计算的 Moran's I 的 Spearman ρ）
    3.5  多方法排名一致性（两两 Spearman ρ / top-K Jaccard / Kendall's W / 共识集）
  维度 4 下游价值（仅当 h5ad obs 有类别标注时）：
    4.1  top-K SVG -> KMeans -> ARI / NMI
    4.3  ARI vs K 特征效率曲线
  维度 7 效率与工程：
    7.1  wall-clock 时间汇总（读各方法 runtime.json）
    7.3  数量-质量权衡散点（x=检出数, y=中位 Moran's I）
    7.4  输出信息量（定性，汇总进 summary.json）

预留接口（后续模拟真值阶段再接）：置换零 FPR / QQ 图 / 模拟注入 TPR-FPR /
marker 回收富集，函数已给出签名与数学定义，当前不被 evaluation 调用。

设计原则：
  * 所有空间自相关统一用 k=6 近邻二元权重（与文档协议一致），且 W 只构建一次。
  * 表达值统一为 library-size normalize + log1p（counts -> log1p）。
  * 所有结果用 numpy/pandas 返回，绘图统一交给 evaluation.py（matplotlib）。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 空间权重与自相关统计量
# ---------------------------------------------------------------------------
def knn_weights(coords: np.ndarray, k: int = 6) -> "sparse.csr_matrix":
    """构造 k 近邻二元对称权重矩阵 W（n x n，对称化后取 0/1）。

    coords: (n, d) 空间坐标。对称化 = union（i->j 或 j->i 任一近邻则 W_ij=1）。
    """
    from sklearn.neighbors import kneighbors_graph

    n = coords.shape[0]
    kk = min(k, n - 1)
    A = kneighbors_graph(coords, n_neighbors=kk, mode="connectivity",
                         include_self=False)
    W = (A + A.T).astype(bool).astype(np.float64)
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def spatial_weights(coords: np.ndarray, k: int = 6, w_def: str = "auto",
                    slice_ids: Optional[Sequence] = None):
    """构造空间权重 W，并返回 ``(W, w_def_label)``。

    coords: (n, d) 空间坐标；d=2 平面，d=3 三维（z 需为物理单位 µm）。
    w_def 取值：
      - "iso"  : 直接在完整坐标上做 k 近邻（2D 或 3D kNN），用于 Stereo-seq 真 3D；
      - "slice": 逐切片 2D kNN 块对角（不建立跨切片边），用于 Slide-seq 未配准期；
      - "auto" : 提供 slice_ids 且 coords 为 3D 时用 "slice"，否则 "iso"。
    slice_ids: 每个 spot 的切片归属（长度 n），仅 "slice" 分支使用。

    返回 (W, label)，label ∈ {"iso_2d","iso_3d","slice"}，写入 summary 以追溯口径。
    """
    from scipy import sparse as sp

    coords = np.asarray(coords, dtype=np.float64)
    n, d = coords.shape
    use_slice = (w_def == "slice") or (
        w_def == "auto" and slice_ids is not None and d > 2)

    if use_slice and slice_ids is not None:
        slice_ids = np.asarray(slice_ids)
        blocks = []
        for sid in pd.unique(slice_ids):
            idx = np.where(slice_ids == sid)[0]
            if len(idx) < 2:
                continue
            sub = coords[idx, :2]                     # 切片内 2D 坐标
            Wsub = knn_weights(sub, k=k)
            gi, gj = Wsub.nonzero()
            blocks.append(sp.coo_matrix(
                (np.ones(len(gi), dtype=np.float64),
                 (idx[gi], idx[gj])), shape=(n, n)))
        W = (sum(blocks).tocsr() if blocks else sp.csr_matrix((n, n)))
        W = (W + W.T).astype(bool).astype(np.float64)
        W.setdiag(0)
        W.eliminate_zeros()
        return W, "slice"

    W = knn_weights(coords, k=k)
    return W, f"iso_{d}d"


def _dense_row(mat, i: int) -> np.ndarray:
    """取 (genes x spots) 矩阵第 i 行，返回稠密 1D numpy（兼容稀疏/稠密）。"""
    from scipy import sparse
    if sparse.issparse(mat):
        return np.asarray(mat[i, :].toarray(), dtype=np.float64).ravel()
    return np.asarray(mat[i], dtype=np.float64)


def _dense_subset(mat, rows=None, cols=None) -> np.ndarray:
    """取矩阵子集并稠密化。rows/cols 为整数索引列表或 None。"""
    from scipy import sparse
    if rows is not None:
        mat = mat[rows, :]
    if cols is not None:
        mat = mat[:, cols]
    if sparse.issparse(mat):
        return mat.toarray()
    return np.asarray(mat, dtype=np.float64)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean()


def morans_i(x: np.ndarray, W) -> float:
    """Moran's I = n/S0 * (z'Wz)/(z'z)，z 为中心化向量，S0 = ΣΣ W_ij。"""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return np.nan
    n = W.shape[0]
    z = _z(x)
    denom = float(z @ z)
    if denom == 0.0 or n == 0:
        return np.nan
    Wz = W @ z
    num = float(z @ Wz)
    S0 = float(W.sum())
    if S0 == 0.0:
        return np.nan
    return (n / S0) * (num / denom)


def gearys_c(x: np.ndarray, W) -> float:
    """Geary's C = (n-1)/(2*S0) * ΣΣ W_ij (x_i - x_j)^2 / Σ (x_i - xbar)^2。

    返回值越接近 0 表示空间正相关越强；上层可用 C* = 1 - C 转成"越大越好"。
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        return np.nan
    n = W.shape[0]
    z = _z(x)
    denom = float(z @ z)
    if denom == 0.0 or n == 0:
        return np.nan
    # ΣΣ W_ij (x_i - x_j)^2 = 2 * (z'Wz 的变体)：用差分公式
    Wc = W.tocoo()
    i, j = Wc.row, Wc.col
    diff_sq = float(np.sum(Wc.data * (x[i] - x[j]) ** 2))
    S0 = float(W.sum())
    if S0 == 0.0:
        return np.nan
    return ((n - 1) / (2 * S0)) * (diff_sq / denom)


def moran_geary_table(expr_mat: np.ndarray, gene_names: Sequence[str],
                      W) -> pd.DataFrame:
    """对 (genes x spots) 表达矩阵逐基因算 Moran's I 与 Geary's C。

    返回 DataFrame(index=gene, columns=[moran_I, geary_C, geary_C_star])，
    其中 geary_C_star = 1 - geary_C（越大越正相关）。
    """
    rows = []
    genes = []
    for g, name in enumerate(gene_names):
        x = _dense_row(expr_mat, g)
        mi = morans_i(x, W)
        gc = gearys_c(x, W)
        genes.append(name)
        rows.append((mi, gc, np.nan if np.isnan(gc) else 1.0 - gc))
    return pd.DataFrame(rows, index=genes,
                        columns=["moran_I", "geary_C", "geary_C_star"])


def null_moran_compare(sig_morans: np.ndarray, all_morans: np.ndarray,
                       n_null: int = 200, seed: int = 0, max_k: int = 500) -> dict:
    """1.8 检出集合 Moran's I 的随机对照置换检验。

    sig_morans: 方法检出基因的 Moran's I 数组。
    all_morans: 全基因已算好的 Moran's I 数组（来自 ``moran_geary_table``）。
    max_k: 检出基因数超过该值时随机抽样子集做 null 对比，避免置换次数 x 基因数
           过大导致超大 spot 数据集上数小时~数十小时的白费核时。
    返回 {median_sig, median_null, mean_null, std_null, effect_z, p_value, n_null, n_sig}。
    其中 effect_z = (median_sig - mean_null)/std_null 为连续效应量：即使所有方法都
    打满置换次数（p 值饱和在 1/(n_null+1)），effect_z 仍能区分方法。
    """
    rng = np.random.default_rng(seed)
    sig_morans = np.asarray(sig_morans, dtype=np.float64)
    sig_morans = sig_morans[np.isfinite(sig_morans)]
    all_morans = np.asarray(all_morans, dtype=np.float64)
    all_morans = all_morans[np.isfinite(all_morans)]

    k = len(sig_morans)
    if k == 0:
        return {"median_sig": np.nan, "median_null": np.nan, "mean_null": np.nan,
                "std_null": np.nan, "effect_z": np.nan,
                "p_value": np.nan, "n_null": n_null, "n_sig": 0}
    if k > max_k:
        sig_morans = sig_morans[rng.choice(k, size=max_k, replace=False)]
        k = max_k

    n_all = len(all_morans)
    if n_all < k:
        k = n_all
    null_medians = np.empty(n_null)
    for r in range(n_null):
        idx = rng.choice(n_all, size=k, replace=False)
        null_medians[r] = np.median(all_morans[idx])
    null_medians = null_medians[np.isfinite(null_medians)]
    med_sig = float(np.nanmedian(sig_morans))
    mean_null = float(np.nanmean(null_medians))
    std_null = float(np.nanstd(null_medians)) if len(null_medians) else np.nan
    effect_z = (med_sig - mean_null) / std_null \
        if (std_null is not None and np.isfinite(std_null) and std_null > 0) else np.nan
    p = (1.0 + float(np.sum(null_medians >= med_sig))) / (1.0 + len(null_medians)) \
        if len(null_medians) else np.nan
    return {"median_sig": med_sig, "median_null": float(np.nanmedian(null_medians)),
            "mean_null": mean_null, "std_null": std_null, "effect_z": effect_z,
            "p_value": p, "n_null": int(len(null_medians)), "n_sig": int(len(sig_morans))}


def signal_quality(expr_mat: np.ndarray, gene_names: Sequence[str],
                   sig_genes: Sequence[str], min_spots: int = 10) -> dict:
    """1.9 检出基因的信号质量检查（技术假阳性代理）。

    对方法检出的基因，统计每个基因「有表达的 spot 数」与「平均表达量」。
    - ``frac_low_spots``：表达 spot 数 < ``min_spots`` 的检出基因占比；高占比说明
      检出大量几乎不表达的基因，疑似技术假阳性（SPARK 论文补充图的做法）。
    - ``median_n_spots`` / ``median_mean_expr``：检出基因表达 spot 数 / 平均表达量的中位数。

    expr_mat: (genes x spots) 的 log1p 表达矩阵（行与 gene_names 对齐）。
    """
    gidx = {g: i for i, g in enumerate(gene_names)}
    idx = [gidx[g] for g in sig_genes if g in gidx]
    if not idx:
        return {"n_sig": 0, "frac_low_spots": np.nan,
                "median_n_spots": np.nan, "median_mean_expr": np.nan}
    sub = _dense_subset(expr_mat, rows=np.asarray(idx, dtype=np.int64))
    n_spots = (sub > 0).sum(axis=1).astype(np.float64)
    mean_expr = sub.mean(axis=1)
    return {"n_sig": int(len(idx)),
            "frac_low_spots": float((n_spots < min_spots).mean()),
            "median_n_spots": float(np.median(n_spots)),
            "median_mean_expr": float(np.median(mean_expr))}


# ---------------------------------------------------------------------------
# 排序/一致性指标
# ---------------------------------------------------------------------------
def spearman_rho(a: Sequence, b: Sequence) -> float:
    """Spearman 秩相关（缺失/非有限值先剔除配对）。"""
    from scipy.stats import spearmanr

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan
    return float(spearmanr(a[mask], b[mask]).correlation)


def top_k_jaccard(a: Sequence, b: Sequence, k: Optional[int] = None) -> float:
    """两个基因排名列表前 k 项的 Jaccard = |A∩B| / |A∪B|。"""
    A = set(a[:k] if k else a)
    B = set(b[:k] if k else b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W 协调系数 = 12 * Σ(R_j - R_bar)^2 / (m^2 (n^3 - n))。

    rank_matrix: (m 方法 x n 基因) 的排名矩阵（rank 从 1 起，越小越显著）。
    """
    m, n = rank_matrix.shape
    if n < 2 or m < 2:
        return np.nan
    R = rank_matrix.sum(axis=0)
    R_bar = R.mean()
    S = float(np.sum((R - R_bar) ** 2))
    denom = (m ** 2) * (n ** 3 - n)
    if denom == 0:
        return np.nan
    return 12.0 * S / denom


def consensus_genes(gene_rank_dfs: dict, min_methods: int = 3,
                    top_k: Optional[int] = None) -> list:
    """多方法共识基因集：至少在 min_methods 个方法的 top_k（或全部）中出现。

    gene_rank_dfs: {method: DataFrame(必须含 'gene' 列，且已按显著性排序)}。
    """
    from collections import Counter

    cnt = Counter()
    for df in gene_rank_dfs.values():
        genes = df["gene"].tolist()
        if top_k is not None:
            genes = genes[:top_k]
        cnt.update(set(genes))
    return [g for g, c in cnt.most_common() if c >= min_methods]


# ---------------------------------------------------------------------------
# 下游价值：top-K SVG -> 聚类 -> ARI/NMI
# ---------------------------------------------------------------------------
def svg_cluster_ari(expr_log: np.ndarray, gene_names: Sequence[str],
                    top_genes: Sequence[str], true_labels: Sequence,
                    n_clusters: int, seed: int = 0, n_pcs: int = 20) -> dict:
    """4.1：用 top-K SVG 的表达做 PCA -> KMeans -> 与真实标注比较 ARI/NMI。

    expr_log: (spots x genes) 的 log1p 表达矩阵（列与 gene_names 对齐）。
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    true_labels = np.asarray(true_labels)
    valid = [g for g in top_genes if g in set(gene_names)]
    if len(valid) == 0:
        return {"ari": np.nan, "nmi": np.nan, "n_genes_used": 0}
    idx = [list(gene_names).index(g) for g in valid]
    X = _dense_subset(expr_log, cols=idx)
    # 标准化到零均值单位方差（PCA 前）
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    X = (X - mu) / sd
    k = min(n_clusters, X.shape[0] - 1, X.shape[1])
    if k < 2:
        return {"ari": np.nan, "nmi": np.nan, "n_genes_used": len(valid)}
    Xp = PCA(n_components=min(n_pcs, X.shape[1])).fit_transform(X)
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xp)
    return {"ari": float(adjusted_rand_score(true_labels, pred)),
            "nmi": float(normalized_mutual_info_score(true_labels, pred)),
            "n_genes_used": len(valid)}


def region_discrimination(expr_log: np.ndarray, gene_names: Sequence[str],
                          top_genes: Sequence[str], labels: Sequence) -> dict:
    """4.4 解剖结构一致性：检出基因区分标注区域的能力（单基因 η² 的中位数）。

    对每个检出基因，在类别标注 ``labels`` 上计算组间平方和占比 η² =
    SS_between / SS_total（∈ [0,1]，越大说明该基因表达越能区分区域），再取所有
    检出基因的中位数。用 η² 而非原始 F 值，好处是取值有界、跨方法可比。

    expr_log: (spots x genes) 的 log1p 表达矩阵（列与 gene_names 对齐）。
    labels: 每个 spot 的类别标注（与 expr_log 行对齐）。
    """
    gidx = {g: i for i, g in enumerate(gene_names)}
    valid = [g for g in top_genes if g in gidx]
    if not valid:
        return {"median_eta2": np.nan, "mean_eta2": np.nan, "n_genes": 0}
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"median_eta2": np.nan, "mean_eta2": np.nan, "n_genes": 0}

    X = _dense_subset(expr_log, cols=[gidx[g] for g in valid])   # spots x k
    grand_means = X.mean(axis=0)
    eta2s = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        ss_total = float(((x - grand_means[j]) ** 2).sum())
        if ss_total == 0.0:
            eta2s[j] = np.nan
            continue
        ss_between = 0.0
        for lab in uniq:
            mask = labels == lab
            xg = x[mask]
            ss_between += float(mask.sum()) * (xg.mean() - grand_means[j]) ** 2
        eta2s[j] = ss_between / ss_total

    eta2s = eta2s[np.isfinite(eta2s)]
    if len(eta2s) == 0:
        return {"median_eta2": np.nan, "mean_eta2": np.nan, "n_genes": len(valid)}
    return {"median_eta2": float(np.median(eta2s)),
            "mean_eta2": float(np.mean(eta2s)),
            "n_genes": int(len(eta2s))}


def random_gene_baseline(expr_mat: np.ndarray, gene_names: Sequence[str],
                         labels, moran_table: pd.DataFrame,
                         top_k_list: Sequence[int], n_clusters: int,
                         seed: int = 0, n_draws: int = 10) -> dict:
    """随机基因基线：随机抽 K 个基因，按与方法同口径计算各项指标，作为对照。

    返回与方法指标同键的 dict，供对比图/表把「Random」当第五个「方法」用：
      - median_moran_I / median_geary_C_star / null_p / rank_vs_moran_rho /
        region_eta2 / frac_low_spots 用 K=top_k_list[0] 个随机基因；
      - ari_top{kk} / nmi_top{kk} 每个 kk 用 kk 个随机基因。
    多次随机（n_draws）取中位数，降低抽样方差。

    期望结果：random 的 Moran/ARI/η² 明显低于真实方法、null_p≈0.5、ρ≈0。
    """
    rng = np.random.default_rng(seed)
    n_genes = len(gene_names)
    all_morans = moran_table["moran_I"].to_numpy(np.float64)
    all_geary_star = moran_table["geary_C_star"].to_numpy(np.float64)
    k0 = min(int(top_k_list[0]), n_genes)
    has_labels = labels is not None

    def _med(v):
        return float(np.nanmedian(np.asarray(v, dtype=np.float64)))

    moran_meds, geary_meds, rho_list, nullp_list, effect_z_list = [], [], [], [], []
    eta2_list, frac_low_list = [], []
    median_n_spots_list, median_mean_expr_list = [], []
    ari_agg = {kk: [] for kk in top_k_list}
    nmi_agg = {kk: [] for kk in top_k_list}

    for _ in range(n_draws):
        idx0 = rng.choice(n_genes, size=k0, replace=False)
        moran_meds.append(_med(all_morans[idx0]))
        geary_meds.append(_med(all_geary_star[idx0]))

        # 随机名次 vs Moran's I（应≈0）
        perm = rng.permutation(n_genes).astype(np.float64) + 1.0
        rho_list.append(spearman_rho(-perm, all_morans))

        # 随机集合 vs 全基因的置换 p（应≈0.5，即不显著）+ 连续效应量
        _null = null_moran_compare(
            all_morans[idx0], all_morans, n_null=50,
            seed=int(rng.integers(0, 2 ** 31)))
        nullp_list.append(_null["p_value"])
        effect_z_list.append(_null["effect_z"])

        genes0 = [gene_names[i] for i in idx0]
        _sq = signal_quality(expr_mat, gene_names, genes0)
        frac_low_list.append(_sq["frac_low_spots"])
        median_n_spots_list.append(_sq["median_n_spots"])
        median_mean_expr_list.append(_sq["median_mean_expr"])
        if has_labels:
            eta2_list.append(region_discrimination(
                expr_mat.T, gene_names, genes0, labels)["median_eta2"])

        for kk in top_k_list:
            kk_c = min(int(kk), n_genes)
            idx = rng.choice(n_genes, size=kk_c, replace=False)
            genes = [gene_names[i] for i in idx]
            if has_labels:
                r = svg_cluster_ari(expr_mat.T, gene_names, genes, labels,
                                    n_clusters=n_clusters, seed=seed)
                ari_agg[kk].append(r["ari"])
                nmi_agg[kk].append(r["nmi"])

    out = {
        "median_moran_I": _med(moran_meds),
        "median_geary_C_star": _med(geary_meds),
        "null_p": _med(nullp_list),
        "rank_vs_moran_rho": _med(rho_list),
        "moran_effect_z": _med(effect_z_list),
        "region_eta2": _med(eta2_list) if eta2_list else np.nan,
        "frac_low_spots": _med(frac_low_list),
        "median_n_spots": _med(median_n_spots_list),
        "median_mean_expr": _med(median_mean_expr_list),
    }
    for kk in top_k_list:
        out[f"ari_top{kk}"] = _med(ari_agg[kk]) if ari_agg[kk] else np.nan
        out[f"nmi_top{kk}"] = _med(nmi_agg[kk]) if nmi_agg[kk] else np.nan
    return out


# ---------------------------------------------------------------------------
# 预留接口（模拟真值阶段再接；当前不调用）
# ---------------------------------------------------------------------------
def fpr_under_permutation_null() -> None:
    """1.1 置换零 FPR：打乱坐标后重跑方法，FPR = 检出数/总基因数。"""
    raise NotImplementedError("需坐标置换 + 方法重跑，留待模拟真值阶段。")


def qq_plot_lambda(pvals: np.ndarray) -> dict:
    """1.2 QQ 图 + 膨胀因子 λ = median(-log10 p) / log10(2)。"""
    pvals = np.asarray(pvals, dtype=np.float64)
    pvals = pvals[(pvals > 0) & (pvals <= 1) & np.isfinite(pvals)]
    if len(pvals) == 0:
        return {"lambda": np.nan}
    lam = float(np.median(-np.log10(pvals)) / np.log10(2.0))
    return {"lambda": lam, "n": len(pvals)}


def kstest_uniform(pvals: np.ndarray) -> float:
    """1.3 置换零 p 值均匀性 KS 检验。"""
    from scipy.stats import kstest

    pvals = np.asarray(pvals, dtype=np.float64)
    pvals = pvals[(pvals >= 0) & (pvals <= 1) & np.isfinite(pvals)]
    if len(pvals) == 0:
        return np.nan
    return float(kstest(pvals, "uniform").pvalue)


def marker_enrichment(sig_genes: Sequence, markers: Sequence,
                      background_size: int) -> dict:
    """1.6 marker 回收富集倍数 + Fisher 精确检验。"""
    from scipy.stats import fisher_exact

    K = len(set(sig_genes))
    M = len(set(markers))
    k = len(set(sig_genes) & set(markers))
    m = background_size
    if K == 0 or m == 0:
        return {"k": k, "FE": np.nan, "fisher_p": np.nan}
    FE = (k / K) / (M / m) if (M / m) > 0 else np.nan
    table = [[k, K - k], [M - k, m - K - (M - k)]]
    table = [[max(0, int(v)) for v in row] for row in table]
    try:
        _, p = fisher_exact(table, alternative="greater")
    except ValueError:
        p = np.nan
    return {"k": k, "FE": float(FE), "fisher_p": float(p)}
