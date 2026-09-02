# -*- coding: utf-8 -*-
"""
h5ad_preprocess.py —— 统一的空间转录组数据预处理脚本（合并版）

将此前散落在多个 py 脚本中的预处理逻辑合并为一个入口，供后续管道化、
批量化调用。核心职责：

  1. 统一读入 Visium / STARmap 等平台的 .h5ad 文件；
  2. 生成 SPARK-X / nnSVG 两个 R 方法**直接可用**的数据类型：
       counts.mtx  (genes x spots，MatrixMarket，原始 counts)
       genes.csv   (基因 symbol，一列、无表头)
       barcodes.csv(spot barcode，一列、无表头)
       location.csv(x / y 空间坐标，行名=barcode)
  3. 留出两个 Python 方法（SpaGCN / SpaSEG）的 h5ad 预处理接口：
       - SpaGCN：写 h5ad 副本 + obs 坐标列(x_array/y_array/x_pixel/y_pixel) + 组织学图像（可选）
       - SpaSEG：写 h5ad 副本 + obsm["spatial"] 坐标

坐标来源（按优先级）：
  1) --spatial 指定的文件（tissue_positions.csv 或 spatial.tar.gz）；
  2) results/local_results/spatial/tissue_positions.csv（已解压的坐标表）；
  3) 与 .h5ad 同目录下的 *_spatial.tar.gz（10x 官方下载包，自动解压读取）；
  4) h5ad.obsm["spatial"]（STARmap 等非 Visium 平台的坐标）。

用法（在项目根目录 F:\\computatinalbiology 下）:
    # 默认：对 Visium_Mouse_Olfactory_Bulb 跑全部四项预处理
    python ./src/preprocess/h5ad_preprocess.py

    # 只生成 R 方法（SPARK-X / nnSVG）数据
    python ./src/preprocess/h5ad_preprocess.py --methods spark nnsvg

    # 自定义数据与输出目录
    python ./src/preprocess/h5ad_preprocess.py \
        --h5ad ./data/Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb.h5ad \
        --spatial ./results/local_results/spatial/tissue_positions.csv \
        --outdir ./results/local_results/Visium_Mouse_Olfactory_Bulb

    # STARmap 小数据（坐标自动取自 obsm["spatial"]），只生成 R 方法数据
    python ./src/preprocess/h5ad_preprocess.py \
        --h5ad ./data/STARmap/mouse_brain_cortex/mouse_brain_STARmap_processed.h5ad \
        --outdir ./results/local_results/mouse_brain_STARmap \
        --methods spark nnsvg
"""
import argparse
import io
import sys
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import io as sio
from scipy import sparse

# ----------------------------------------------------------------------
# 路径常量（本文件位于 src/preprocess/ 下，parents[2] = 项目根）
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "local_results"

# 默认数据集（Visium_Mouse_Olfactory_Bulb，用于跑通与批量化模板）
DEFAULT_H5AD = DATA_DIR / "Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb.h5ad"
DEFAULT_SPATIAL_TGZ = DATA_DIR / "Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb_spatial.tar.gz"
DEFAULT_OUTDIR = RESULTS_DIR / "Visium_Mouse_Olfactory_Bulb"

# STARmap 小数据集（坐标位于 obsm["spatial"]，适合快速测试）
STARMAP_H5AD = DATA_DIR / "STARmap/mouse_brain_cortex/mouse_brain_STARmap_processed.h5ad"
STARMAP_OUTDIR = RESULTS_DIR / "mouse_brain_STARmap"

# 四种方法的输出子目录（与 results/local_results/Visium_Mouse_Olfactory_Bulb/ 对齐）
SUBDIRS = {"spark": "SPARK_X", "nnsvg": "nnSVG", "spagcn": "spaGCN", "spaseg": "spaSEG"}


# ----------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------
def _read_tissue_positions(source: Path) -> pd.DataFrame:
    """读取 tissue_positions.csv（或其所在的 spatial.tar.gz），返回以 barcode 为索引的 DataFrame。"""
    if source.suffix == ".gz" or source.suffix == ".tgz" or ".tar" in source.name:
        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_positions.csv")), None)
            if member is None:
                raise FileNotFoundError(f"{source} 内未找到 tissue_positions.csv")
            raw = tar.extractfile(member).read()
        tp = pd.read_csv(io.BytesIO(raw))
    else:
        tp = pd.read_csv(source)
    if "barcode" not in tp.columns:
        raise ValueError("tissue_positions.csv 缺少 barcode 列")
    return tp.set_index("barcode")


def _read_hires_image(source: Path):
    """从 spatial.tar.gz 提取 tissue_hires_image.png（组织学图像），无则返回 None。"""
    try:
        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_hires_image.png")), None)
            if member is None:
                return None
            data = tar.extractfile(member).read()
        import cv2  # 延迟导入，避免不必要依赖

        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR 数组（SpaGCN 官方用法）
    except Exception:
        return None


def _resolve_spatial(h5ad_path: Path, spatial_arg) -> tuple:
    """按优先级解析 (tissue_positions.csv 路径, spatial.tar.gz 路径)。"""
    # 1) 用户显式指定
    if spatial_arg is not None:
        s = Path(spatial_arg)
        if s.exists():
            if s.suffix == ".csv":
                return s, None
            return None, s
    # 2) 已解压的坐标表
    extracted = RESULTS_DIR / "spatial" / "tissue_positions.csv"
    if extracted.exists():
        return extracted, None
    # 3) h5ad 同目录下的 *_spatial.tar.gz
    tgz = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
    if tgz.exists():
        return None, tgz
    # 4) 默认数据集的 spatial.tar.gz
    if DEFAULT_SPATIAL_TGZ.exists():
        return None, DEFAULT_SPATIAL_TGZ
    return None, None


def _load_coords(h5ad: ad.AnnData, spatial_arg) -> pd.DataFrame:
    """读入坐标并与 h5ad 的 obs.index 对齐，返回统一列结构的 DataFrame（索引=barcode）。

    Visium：列含 x(=pxl_col_in_fullres)/y(=pxl_row_in_fullres)，若存在则保留 array_row/array_col；
    STARmap 等：回退到 obsm["spatial"]，x/y 为空间坐标前两列。
    """
    csv_path, tgz_path = _resolve_spatial(Path(h5ad.filename) if h5ad.filename else DEFAULT_H5AD,
                                          spatial_arg)
    tp = None
    if csv_path is not None:
        tp = _read_tissue_positions(csv_path)
    elif tgz_path is not None:
        tp = _read_tissue_positions(tgz_path)

    if tp is not None:
        # Visium：默认只保留 in_tissue=1 的 spot，并统一列名
        if "in_tissue" in tp.columns:
            tp = tp[tp["in_tissue"] == 1]
        tp = tp.loc[[b for b in h5ad.obs.index if b in tp.index]]
        if "pxl_col_in_fullres" in tp.columns and "pxl_row_in_fullres" in tp.columns:
            tp = tp.rename(columns={"pxl_col_in_fullres": "x", "pxl_row_in_fullres": "y"})
        # 若坐标表与 h5ad barcode 无交集（如 STARmap 误命中 Visium 全局坐标表），
        # 则回退到 obsm["spatial"]，避免对齐后为空
        if len(tp) > 0:
            return tp

    # STARmap 等非 Visium 平台：直接用 obsm["spatial"]
    if "spatial" in h5ad.obsm and h5ad.obsm["spatial"] is not None:
        coords = np.asarray(h5ad.obsm["spatial"], dtype=float)
        return pd.DataFrame(coords[:, :2], index=h5ad.obs.index, columns=["x", "y"])

    raise FileNotFoundError("找不到坐标来源（tissue_positions.csv / spatial.tar.gz / obsm['spatial']）")


# ----------------------------------------------------------------------
# 1) R 方法（SPARK-X / nnSVG）——导出通用矩阵格式
# ----------------------------------------------------------------------
def export_r_format(h5ad: ad.AnnData, tp: pd.DataFrame, outdir: Path) -> None:
    """导出 SPARK-X / nnSVG 可直接读取的 counts.mtx / genes.csv / barcodes.csv / location.csv。

    counts.mtx 为 genes x spots（行为基因），值与原始 counts 一致。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    X = h5ad.X
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.astype(np.float64)

    gene_names = list(h5ad.var.index.astype(str))

    # 与坐标对齐（保持顺序一致）
    barcodes = [b for b in h5ad.obs.index if b in tp.index]
    keep_idx = [i for i, b in enumerate(h5ad.obs.index) if b in tp.index]
    X = X[keep_idx, :]                      # spots x genes
    tp_aligned = tp.loc[barcodes]

    # 转置为 genes x spots（R 方法需要行为基因）
    X = X.T.tocsr()

    sio.mmwrite(str(outdir / "counts.mtx"), X, field="integer")
    pd.Series(gene_names).to_csv(outdir / "genes.csv", index=False, header=False)
    pd.Series(barcodes).to_csv(outdir / "barcodes.csv", index=False, header=False)

    loc = pd.DataFrame({
        "x": tp_aligned["x"].values,
        "y": tp_aligned["y"].values,
    }, index=barcodes)
    loc.to_csv(outdir / "location.csv", index_label="barcode")

    print(f"      [R 格式] {X.shape[0]} genes x {X.shape[1]} spots -> {outdir}")


# ----------------------------------------------------------------------
# 2) SpaGCN 预处理接口
# ----------------------------------------------------------------------
def prepare_spagcn(h5ad: ad.AnnData, tp: pd.DataFrame, img, outdir: Path,
                   sample_name: str = "Visium_Mouse_Olfactory_Bulb") -> None:
    """为 SpaGCN 准备输入：写 h5ad 副本 + 坐标列 + 组织学图像。

    SpaGCN 官方要求的 obs 坐标列（对应 positions.txt 的 6 列）：
      x_array = array_row,  y_array = array_col
      x_pixel = pxl_row_in_fullres, y_pixel = pxl_col_in_fullres
    组织学图像可选（histology=False 时 SpaGCN 仅用空间坐标建图）。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    # 复制一份，避免污染原始对象
    adata = ad.AnnData(
        X=h5ad.X.copy(),
        obs=h5ad.obs.copy(),
        var=h5ad.var.copy(),
        obsm={k: v.copy() for k, v in h5ad.obsm.items()},
        uns={k: v for k, v in h5ad.uns.items()},
    )
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    adata.obs["x_pixel"] = tp_aligned["x"].astype(float).values
    adata.obs["y_pixel"] = tp_aligned["y"].astype(float).values
    if "array_row" in tp_aligned.columns and "array_col" in tp_aligned.columns:
        adata.obs["x_array"] = tp_aligned["array_row"].astype(int).values
        adata.obs["y_array"] = tp_aligned["array_col"].astype(int).values
    else:
        # 非 Visium 平台无 array_row/array_col，退化为像素坐标取整
        adata.obs["x_array"] = tp_aligned["x"].astype(int).values
        adata.obs["y_array"] = tp_aligned["y"].astype(int).values

    h5ad_out = outdir / f"{sample_name}_spaGCN.h5ad"
    adata.write(h5ad_out)

    img_out = outdir / "histology.png"
    if img is not None:
        import cv2

        cv2.imwrite(str(img_out), img)
        print(f"      [SpaGCN] 坐标列 + 组织学图像 -> {h5ad_out} / {img_out}")
    else:
        print(f"      [SpaGCN] 坐标列（无组织学图像，可用 histology=False）-> {h5ad_out}")


# ----------------------------------------------------------------------
# 3) SpaSEG 预处理接口
# ----------------------------------------------------------------------
def prepare_spaseg(h5ad: ad.AnnData, tp: pd.DataFrame, outdir: Path,
                   sample_name: str = "Visium_Mouse_Olfactory_Bulb") -> None:
    """为 SpaSEG 准备输入：写 h5ad 副本 + obsm["spatial"] 坐标。

    SpaSEG 通过 scanpy 读取 h5ad，并用 obsm["spatial"] 获取空间坐标。
    """
    outdir.mkdir(parents=True, exist_ok=True)

    adata = ad.AnnData(
        X=h5ad.X.copy(),
        obs=h5ad.obs.copy(),
        var=h5ad.var.copy(),
        obsm={k: v.copy() for k, v in h5ad.obsm.items()},
        uns={k: v for k, v in h5ad.uns.items()},
    )
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    # obsm["spatial"]：(n_spots, 2)，列为 x / y
    adata.obsm["spatial"] = np.column_stack([
        tp_aligned["x"].astype(float).values,
        tp_aligned["y"].astype(float).values,
    ]).astype(np.float32)

    h5ad_out = outdir / f"{sample_name}_spaSEG.h5ad"
    adata.write(h5ad_out)
    print(f"      [SpaSEG] obsm['spatial'] -> {h5ad_out}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="统一 h5ad 预处理：生成 SPARK-X/nnSVG 数据 + SpaGCN/SpaSEG 接口"
    )
    ap.add_argument("--h5ad", type=str, default=None,
                    help=f"输入 h5ad 路径（默认 {DEFAULT_H5AD.name}）")
    ap.add_argument("--spatial", type=str, default=None,
                    help="tissue_positions.csv 或 spatial.tar.gz（默认自动查找）")
    ap.add_argument("--outdir", type=str, default=None,
                    help="输出根目录（默认 results/local_results/Visium_Mouse_Olfactory_Bulb）")
    ap.add_argument("--methods", nargs="+", choices=["spark", "nnsvg", "spagcn", "spaseg"],
                    default=["spark", "nnsvg", "spagcn", "spaseg"],
                    help="要生成的数据（默认全部四项）")
    ap.add_argument("--sample-name", type=str, default=None,
                    help="样本名（用于 SpaGCN/SpaSEG 输出文件名前缀，默认取 h5ad 文件名）")
    args = ap.parse_args()

    # ---- 解析路径 ----
    h5ad_path = Path(args.h5ad) if args.h5ad else DEFAULT_H5AD
    if not h5ad_path.exists():
        cand = DATA_DIR / h5ad_path
        if cand.exists():
            h5ad_path = cand
    if not h5ad_path.exists():
        print(f"ERROR: 找不到 h5ad 文件: {h5ad_path}")
        sys.exit(1)
    out_root = Path(args.outdir) if args.outdir else DEFAULT_OUTDIR
    sample_name = args.sample_name or h5ad_path.stem

    # ---- 读取数据与坐标 ----
    print(f"[1/3] 读取 h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"      shape = {adata.shape} (spots x genes)")

    print("[2/3] 解析空间坐标（in_tissue only）")
    tp = _load_coords(adata, args.spatial)
    print(f"      对齐后 spots = {len(tp)}")

    img = None
    if "spagcn" in args.methods:
        # 组织学图像仅存在于 spatial.tar.gz 中，独立查找（不受 csv 优先级影响）
        tgz_path = None
        if args.spatial is not None and Path(args.spatial).exists():
            s = Path(args.spatial)
            if s.suffix != ".csv":
                tgz_path = s
        else:
            tgz_path = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
            if not tgz_path.exists():
                tgz_path = DEFAULT_SPATIAL_TGZ
        if tgz_path is not None and tgz_path.exists():
            img = _read_hires_image(tgz_path)
            if img is not None:
                print("      已提取组织学图像 (tissue_hires_image.png)")

    # ---- 按方法生成 ----
    print("[3/3] 生成各方法输入")
    for m in args.methods:
        sub = out_root / SUBDIRS[m]
        if m in ("spark", "nnsvg"):
            export_r_format(adata, tp, sub)
        elif m == "spagcn":
            prepare_spagcn(adata, tp, img, sub, sample_name=sample_name)
        elif m == "spaseg":
            prepare_spaseg(adata, tp, sub, sample_name=sample_name)

    print("===== h5ad_preprocess 完成 ✓ =====")


if __name__ == "__main__":
    main()
