# -*- coding: utf-8 -*-
"""
将 Visium h5ad 导出为 R 可读的中间文件（供 SPARK / nnSVG 使用）。

输出到 results/local_results/Visium_Mouse_Olfactory_Bulb/SPARK_X/：
  - counts.mtx        : 稀疏矩阵 (genes x spots)，MatrixMarket 格式，值为原始 counts
  - genes.csv         : 行名（基因 symbol）
  - barcodes.csv      : 列名（spot barcode）
  - location.csv      : 坐标 (x = pxl_col_in_fullres, y = pxl_row_in_fullres)

用法：
  python export_h5ad_to_r.py --h5ad <path> --outdir <dir> --spatial <tissue_positions.csv>
"""
import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io, sparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--spatial", required=True, help="tissue_positions.csv")
    ap.add_argument("--in-tissue-only", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[1/4] 读取 h5ad: {args.h5ad}")
    adata = ad.read_h5ad(args.h5ad)
    print(f"      shape = {adata.shape}")

    # X 为 counts（float32 稀疏），形状 = spots x genes (n_obs x n_vars)
    X = adata.X
    if sparse.issparse(X):
        X = X.astype(np.float64)
    else:
        X = sparse.csr_matrix(X.astype(np.float64))

    # 基因名：var.index（symbol）
    gene_names = list(adata.var.index.astype(str))

    # spot barcode：obs.index
    barcodes = list(adata.obs.index.astype(str))

    # 坐标：从 tissue_positions.csv 读取，与 h5ad barcode 对齐
    print(f"[2/4] 读取坐标: {args.spatial}")
    tp = pd.read_csv(args.spatial)
    tp = tp.set_index("barcode")
    # 默认只保留组织内 spot
    if args.in_tissue_only and "in_tissue" in tp.columns:
        tp = tp[tp["in_tissue"] == 1]
    tp = tp.loc[[b for b in barcodes if b in tp.index]]

    # 对齐：只保留有坐标的 spot（行索引）
    keep_cols = [b for b in barcodes if b in tp.index]
    keep_idx = [i for i, b in enumerate(barcodes) if b in keep_cols]
    X = X[keep_idx, :]          # 过滤 spot（行）
    barcodes = keep_cols
    tp = tp.loc[keep_cols]
    loc = tp[["pxl_col_in_fullres", "pxl_row_in_fullres"]].rename(
        columns={"pxl_col_in_fullres": "x", "pxl_row_in_fullres": "y"}
    )
    # 转置为 genes x spots（SPARK / nnSVG 需要行为基因）
    X = X.T.tocsr()
    print(f"      对齐后 shape = {X.shape} (genes x spots)")

    # X 是 genes x spots，保持行=基因
    print("[3/4] 写 counts.mtx (genes x spots)")
    io.mmwrite(os.path.join(args.outdir, "counts.mtx"), X, field="integer")

    pd.Series(gene_names).to_csv(os.path.join(args.outdir, "genes.csv"),
                                 index=False, header=False)
    pd.Series(barcodes).to_csv(os.path.join(args.outdir, "barcodes.csv"),
                               index=False, header=False)
    loc.to_csv(os.path.join(args.outdir, "location.csv"), index_label="barcode")

    print("[4/4] 完成。输出文件：")
    for f in ["counts.mtx", "genes.csv", "barcodes.csv", "location.csv"]:
        print(f"      {os.path.join(args.outdir, f)}")


if __name__ == "__main__":
    main()
