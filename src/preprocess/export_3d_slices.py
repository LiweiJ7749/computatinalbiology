# -*- coding: utf-8 -*-
"""export_3d_slices.py —— 把 dim=3 数据逐切片导出为 2D 方法（nnSVG/SpaGCN/SpaSEG）输入。

背景：SPARK-X 走原生 3D（见 src._preprocess_run_3d / run_spark.r）；而 nnSVG/SpaGCN/SpaSEG
受 2D 底层假设约束，只能在 3D 数据上做“逐切片 2D 检测 + 跨切片合并”（docs/3d_svg_detection.md）。

本脚本只负责“逐切片生成 2D 输入”，运行与合并分别由
src/pipeline/run_3d_benchmark.sh 与 src/py_models/merge_slices.py 完成：

  - Slide-seq : 每个切片文件即 1 片（run["slices"]）
  - Stereo-seq: 单文件按 obs['slice_ID']（回退 spatial 第 3 列 z）切分

输出目录约定（sample 后缀加 _S<i>）：
  nnSVG : <outdir>/nnSVG/slices/S<i>/           counts.mtx/genes.csv/barcodes.csv/location.csv
  SpaGCN: <outdir>/spaGCN/slices/S<i>/spaGCN/   <sample>_S<i>_spaGCN.h5ad
  SpaSEG: <outdir>/spaSEG/slices/S<i>/spaSEG/   <sample>_S<i>_spaSEG.h5ad

用法（项目根，envs/spatial 的 python）：
  python src/preprocess/export_3d_slices.py --dataset Slide_seq_OB2_3D --methods nnsvg spagcn spaseg
  python src/preprocess/export_3d_slices.py --dataset Stereo_seq_drosophila --methods nnsvg
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402

TWO_D_METHODS = ("nnsvg", "spagcn", "spaseg")


def _resolve(args):
    """解析 run（dim=3）。用 methods=["spark"] 规避 resolve_run 对 2D 方法的 3D 过滤。"""
    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad, outdir=args.outdir,
                          sample=args.sample, methods=["spark"])
    if int(run.get("dim") or 2) != 3:
        raise SystemExit(f"数据集 {args.dataset or args.h5ad} 不是 dim=3（dim={run.get('dim')}）")
    return run


def _slice_dir(run, method, i):
    """返回方法 m 的第 i 片数据目录。"""
    sub = src.METHOD_SUBDIRS[method]
    return run["outdir"] / sub / "slices" / f"S{i}" / (sub if method != "nnsvg" else "")


def _export_one(adata, tp, run, method, i, sample_slice):
    """把单个 2D 切片导出为方法 method 的输入。"""
    if method == "nnsvg":
        outdir = _slice_dir(run, "nnsvg", i)
        src.export_r_format(adata, tp, outdir, tech=run.get("tech"), dim=2)
    elif method == "spagcn":
        outdir = _slice_dir(run, "spagcn", i)
        src.prepare_spagcn(adata, tp, None, outdir, sample_slice)
    elif method == "spaseg":
        outdir = _slice_dir(run, "spaseg", i)
        src.prepare_spaseg(adata, tp, outdir, sample_slice)


def _tp_from_subset(subset, tech):
    """从子集 AnnData 生成 2D 坐标 DataFrame（index=barcode, 列 x/y）。"""
    return src.load_coords(subset, None, tech=tech, h5ad_path=None, dim=2)


def export_slideseq(run, methods):
    """Slide-seq：每个切片文件即一片。"""
    import anndata as ad

    for i, p in enumerate(run["slices"]):
        p = Path(p)
        adata = ad.read_h5ad(p)
        tp = src.load_coords(adata, None, tech=run.get("tech"), h5ad_path=p, dim=2)
        sample_slice = f"{run['sample']}_S{i}"
        src.log_message(f"[Slide-seq 切片 {i}] {p.name}: {adata.shape[0]} spots")
        for m in methods:
            _export_one(adata, tp, run, m, i, sample_slice)
        del adata


def export_stereoseq(run, methods):
    """Stereo-seq：单文件按 slice_ID 切分。"""
    import anndata as ad

    adata = ad.read_h5ad(run["h5ad"])
    ids = src.stereo_slice_ids(adata)
    src.log_message(f"[Stereo-seq] 共 {len(ids)} 个切片 id: {ids}")
    for i, sid in enumerate(ids):
        mask = adata.obs["slice_ID"].astype(str).to_numpy() == sid \
            if "slice_ID" in adata.obs.columns else \
            np.asarray(adata.obsm["spatial"], dtype=float)[:, 2] == float(sid)
        barcodes = [b for b, k in zip(adata.obs.index, mask) if k]
        subset = src._subset_adata(adata, barcodes)
        tp = _tp_from_subset(subset, run.get("tech"))
        sample_slice = f"{run['sample']}_S{i}"
        src.log_message(f"[Stereo-seq 切片 {i}/{sid}] {subset.shape[0]} spots")
        for m in methods:
            _export_one(subset, tp, run, m, i, sample_slice)
        del subset
    del adata


def main():
    ap = argparse.ArgumentParser(description="3D 数据逐切片导出 2D 方法输入")
    ap.add_argument("--dataset", default=None, help="dim=3 数据集 key")
    ap.add_argument("--h5ad", default=None, help="显式 h5ad（Slide-seq 用 --dataset 的 slices）")
    ap.add_argument("--outdir", default=None, help="输出根目录（默认 results/local_results/<dataset>）")
    ap.add_argument("--sample", default=None, help="样本标签")
    ap.add_argument("--methods", nargs="+", choices=list(TWO_D_METHODS),
                    default=list(TWO_D_METHODS), help="要导出的 2D 方法")
    args = ap.parse_args()

    run = _resolve(args)
    src.log_header(f"3D 逐切片导出: {run['dataset']}")
    src.log_message(f"outdir = {run['outdir']} | tech = {run['tech']} | "
                    f"methods = {args.methods}")

    if run.get("tech") == "Slide_seq":
        export_slideseq(run, args.methods)
    elif run.get("tech") == "Stereo_seq":
        export_stereoseq(run, args.methods)
    else:
        raise SystemExit(f"技术类型 {run.get('tech')!r} 尚无逐切片导出实现（仅 Slide_seq/Stereo_seq）")

    src.log_message("逐切片导出完成", section="完成")


if __name__ == "__main__":
    main()
