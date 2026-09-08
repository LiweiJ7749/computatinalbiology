"""Visium HD (Space Ranger 3.x) feature_slice.h5 -> h5ad 转换脚本。

将 Visium HD 的 feature_slice.h5（Space Ranger 3.x 新格式，无 matrix 组）
与 spatial.tar.gz（仅含图像，无坐标）转换为通用 .h5ad 文件，供后续
SVG 检测方法对比使用。

说明：
    - 表达计数与空间坐标都来自 feature_slice.h5 内部：
        feature_slices/{gene_index}/row,col,data
      row/col 为 2um 像素坐标，data 为 UMI 计数。
    - 聚合到 bin：2um 像素 //(bin_size/2) -> bin grid。
      bin_size=8 -> 8um bin，bin_size=16 -> 16um bin。
    - 仅保留组织内 bin（masks/square_XXXum）。
    - spatial.tar.gz 里的 H&E 图像提取后存入 adata.uns['spatial'] 供可视化。
    - 逐基因聚合相互独立，支持多进程并行（--n-jobs），以榨取多核 CPU；
      内存占用主要与 16um 下的非零元总数相关，bin_size=16 约为 8um 的 1/4。

依赖（envs/spatial 已装）：h5py / numpy / scipy / scanpy / matplotlib。

用法（项目根目录）:
    python ./src/preprocess/10xVisium_pretreat_h5toh5ad.py \\
        ./data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_feature_slice.h5 \\
        ./data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_spatial.tar.gz \\
        ./data/10xVisium/Human_Breast_Cancer/Visium_HD_Human_Breast_Cancer.h5ad \\
        [--bin-size 16] [--n-jobs 24]
"""
import argparse
import os
import sys
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import anndata as ad


# ---------------------------------------------------------------------------
# 逐基因聚合（供单进程与多进程 worker 复用）
# ---------------------------------------------------------------------------
def _process_gene(fs, key: str, bf: int, max_r: int, max_c: int,
                  grid: np.ndarray, n_bins: int):
    """把单个基因的 2um 像素 UMI 聚合到 bin，返回 (cols, vals)（CSR 一行的非零列/值）。"""
    g = fs[key]
    rr = g["row"][:] // bf
    cc = g["col"][:] // bf
    keep = (rr <= max_r) & (cc <= max_c)
    rr = rr[keep]
    cc = cc[keep]
    if len(rr) == 0:
        return np.empty(0, np.int32), np.empty(0, np.float32)
    colidx = grid[rr, cc]
    good = colidx >= 0
    if not good.any():
        return np.empty(0, np.int32), np.empty(0, np.float32)
    data = g["data"][:][keep][good].astype(np.float32)
    counts = np.bincount(colidx[good], weights=data, minlength=n_bins)
    nz = np.nonzero(counts)[0]
    return nz.astype(np.int32), counts[nz].astype(np.float32)


def _aggregate_serial(h5_path, gene_keys, bf, max_r, max_c, grid, n_bins):
    cols_all, vals_all = [], []
    with h5py.File(h5_path, "r") as f:
        fs = f["feature_slices"]
        for key in gene_keys:
            c, v = _process_gene(fs, key, bf, max_r, max_c, grid, n_bins)
            cols_all.append(c)
            vals_all.append(v)
    return cols_all, vals_all


# 多进程 worker 的模块级上下文（每个 worker 进程打开自己的 h5py 只读句柄）
_WORKER_CTX = None


def _init_worker(h5_path, bf, max_r, max_c, grid, n_bins):
    global _WORKER_CTX
    _WORKER_CTX = (str(h5_path), int(bf), int(max_r), int(max_c),
                   grid, int(n_bins))


def _run_chunk(gene_keys):
    h5_path, bf, max_r, max_c, grid, n_bins = _WORKER_CTX
    cols_all, vals_all = [], []
    with h5py.File(h5_path, "r") as f:
        fs = f["feature_slices"]
        for key in gene_keys:
            c, v = _process_gene(fs, key, bf, max_r, max_c, grid, n_bins)
            cols_all.append(c)
            vals_all.append(v)
    return cols_all, vals_all


def _aggregate_parallel(h5_path, gene_keys, bf, max_r, max_c, grid, n_bins,
                        n_jobs):
    n = len(gene_keys)
    n_jobs = max(1, min(int(n_jobs), n))
    chunk_size = (n + n_jobs - 1) // n_jobs
    chunks = [gene_keys[i:i + chunk_size] for i in range(0, n, chunk_size)]
    with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker,
            initargs=(h5_path, bf, max_r, max_c, grid, n_bins)) as ex:
        results = list(ex.map(_run_chunk, chunks))
    cols_all = [c for cols, _ in results for c in cols]
    vals_all = [v for _, vals in results for v in vals]
    return cols_all, vals_all


def build_adata(h5_path: Path, bin_size: int, n_jobs: int = 1) -> ad.AnnData:
    bf = bin_size // 2  # 2um 像素 -> bin 的除数
    bin_key = f"square_{bin_size:03d}um"

    with h5py.File(h5_path, "r") as f:
        # ---- 基因信息 ----
        feat = f["features"]
        names = feat["name"][:].astype(str)
        ids = feat["id"][:].astype(str)
        types = feat["feature_type"][:].astype(str)
        genome = feat["genome"][:].astype(str)

        # ---- 组织内 bin 掩码 -> 列索引映射 ----
        mask = f[f"masks/{bin_key}"]
        pairs = np.unique(
            np.stack([mask["row"][:], mask["col"][:]], axis=1), axis=0
        )
        grid = np.full(
            (int(pairs[:, 0].max()) + 1, int(pairs[:, 1].max()) + 1),
            -1, dtype=np.int32,
        )
        grid[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs), dtype=np.int32)
        bin_rows = pairs[:, 0].astype(np.int32)
        bin_cols = pairs[:, 1].astype(np.int32)
        n_bins = len(pairs)
        max_r, max_c = grid.shape[0] - 1, grid.shape[1] - 1

        gene_keys = sorted(f["feature_slices"].keys(), key=int)

    gene_idx = np.array([int(k) for k in gene_keys], dtype=np.int64)

    # ---- 逐基因聚合 UMI 到 bin（单进程或多进程） ----
    if n_jobs > 1 and len(gene_keys) > n_jobs:
        cols_all, vals_all = _aggregate_parallel(
            h5_path, gene_keys, bf, max_r, max_c, grid, n_bins, n_jobs)
    else:
        cols_all, vals_all = _aggregate_serial(
            h5_path, gene_keys, bf, max_r, max_c, grid, n_bins)

    # ---- 组装 CSR（行=基因，列=bin），再转置为 bins x genes ----
    indptr = np.zeros(len(gene_keys) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(c) for c in cols_all])
    indices = np.concatenate(cols_all).astype(np.int32)
    data = np.concatenate(vals_all).astype(np.float32)
    X_genes_bins = csr_matrix(
        (data, indices, indptr), shape=(len(gene_keys), n_bins)
    )
    X = X_genes_bins.T.tocsr()  # bins x genes

    # ---- var / obs ----
    var = pd.DataFrame(
        {"gene_ids": ids[gene_idx], "feature_types": types[gene_idx],
         "genome": genome[gene_idx]},
        index=names[gene_idx],
    )
    if not var.index.is_unique:
        var.index = pd.Index(
            [f"{n}_{i}" if var.index.duplicated(keep=False)[i] else n
             for i, n in enumerate(var.index)]
        )

    obs = pd.DataFrame(
        {"array_row": bin_rows, "array_col": bin_cols},
        index=[f"{r}-{c}" for r, c in zip(bin_rows, bin_cols)],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    # 空间坐标（um）：bin 中心 = (grid + 0.5) * bin_size
    adata.obsm["spatial"] = np.column_stack(
        [(bin_cols + 0.5) * bin_size, (bin_rows + 0.5) * bin_size]
    ).astype(np.float32)
    adata.uns["spatial"] = {
        bin_key: {"scalefactors": {"tissue_hires_scalef": 1.0}}
    }
    return adata


def load_image(spatial_tgz: Path, key: str, out_dir: str):
    """从 spatial.tar.gz 提取指定图像为 numpy 数组，失败返回 None。"""
    try:
        with tarfile.open(spatial_tgz, "r:gz") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith(key)), None)
            if member is None:
                return None
            tar.extract(member, path=out_dir)
        from matplotlib import image as mpimg

        return mpimg.imread(str(Path(out_dir) / member.name))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Visium HD feature_slice.h5 -> h5ad")
    ap.add_argument("h5", help="feature_slice.h5 路径")
    ap.add_argument("spatial_tgz", help="spatial.tar.gz 路径")
    ap.add_argument("output", help="输出 h5ad 路径")
    ap.add_argument("--bin-size", type=int, default=8, choices=[8, 16],
                    help="bin 大小（um），默认 8；大数据集用 16 控制规模")
    ap.add_argument("--n-jobs", type=int,
                    default=(os.cpu_count() or 1),
                    help="并行进程数（默认=CPU 核数；1=单进程）")
    args = ap.parse_args()

    h5 = Path(args.h5)
    if not h5.exists():
        print(f"ERROR: 找不到 {h5}")
        sys.exit(1)

    print(f"读取 {h5.name}（bin_size={args.bin_size}um, n_jobs={args.n_jobs}）...")
    adata = build_adata(h5, args.bin_size, args.n_jobs)

    tgz = Path(args.spatial_tgz)
    if tgz.exists():
        with tempfile.TemporaryDirectory() as tmp:
            hires = load_image(tgz, "tissue_hires_image.png", tmp)
            lowres = load_image(tgz, "tissue_lowres_image.png", tmp)
            if hires is not None or lowres is not None:
                lib = f"square_{args.bin_size:03d}um"
                imgs = {}
                if hires is not None:
                    imgs["hires"] = hires
                if lowres is not None:
                    imgs["lowres"] = lowres
                adata.uns["spatial"][lib]["images"] = imgs
                print("已提取空间图像")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out)
    print(
        f"OK: {out} ({adata.n_obs} bins x {adata.n_vars} genes, "
        f"spatial {adata.obsm['spatial'].shape})"
    )


if __name__ == "__main__":
    main()
