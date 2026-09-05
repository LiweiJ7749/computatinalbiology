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
        x = expr_mat[g]
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
    返回 {median_sig, median_null, mean_null, p_value(单侧, sig>null), n_null, n_sig}。
    """
    rng = np.random.default_rng(seed)
    sig_morans = np.asarray(sig_morans, dtype=np.float64)
    sig_morans = sig_morans[np.isfinite(sig_morans)]
    all_morans = np.asarray(all_morans, dtype=np.float64)
    all_morans = all_morans[np.isfinite(all_morans)]

    k = len(sig_morans)
    if k == 0:
        return {"median_sig": np.nan, "median_null": np.nan, "mean_null": np.nan,
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
    p = (1.0 + float(np.sum(null_medians >= med_sig))) / (1.0 + len(null_medians)) \
        if len(null_medians) else np.nan
    return {"median_sig": med_sig, "median_null": float(np.nanmedian(null_medians)),
            "mean_null": float(np.nanmean(null_medians)), "p_value": p,
            "n_null": int(len(null_medians)), "n_sig": int(len(sig_morans))}


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
    X = expr_log[:, idx]
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
