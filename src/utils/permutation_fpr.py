#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""permutation_fpr.py —— 坐标置换 FPR（金标准假阳性率）

把空间坐标随机打乱后重跑方法，统计“零假设（坐标与表达无关）下方法仍报显著”的
基因比例：FPR = mean(n_sig) / n_genes。理想情况下 FPR ≈ α（0.05）。

当前实现：
  - spark  : SPARK-X 原生 3D，秒级，可本地/单节点直接跑。
  - spagcn : 逐切片 2D + 跨切片合并，需重跑每片训练，代价大，HPC job array 并行。
  - spaseg : 逐切片 2D + 跨切片合并，同上。

用法：
  # 顺序跑 N 次并汇总（本地；spark 快，spagcn/spaseg 建议 HPC）
  python src/utils/permutation_fpr.py --dataset Stereo_seq_drosophila --method spark --n-perms 10

  # HPC job array：每个数组任务跑一次置换
  python src/utils/permutation_fpr.py --dataset Stereo_seq_drosophila --method spagcn --perm-idx 0 --seed 0

  # 汇总已有 perm_*.json
  python src/utils/permutation_fpr.py --dataset Stereo_seq_drosophila --method spagcn --collect
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402


def _r_env():
    env = dict(os.environ)
    rbin = Path(src.find_rscript())
    rlib = rbin.parent.parent / "lib"
    env["PATH"] = str(rbin.parent) + os.pathsep + env.get("PATH", "")
    if rlib.exists():
        env["LD_LIBRARY_PATH"] = str(rlib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return str(rbin), env


def _run_sparkx(data_dir: Path, sample: str):
    rscript, env = _r_env()
    src.log_message(f"运行 SPARK-X 于 {data_dir}")
    p = subprocess.run(
        [rscript, str(ROOT / "src/r_models/run_spark.r"), str(data_dir), sample],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if p.returncode != 0:
        raise RuntimeError(f"SPARK-X 返回非零: {p.returncode}")
    rank = pd.read_csv(data_dir / f"SVG_SPARK_{sample}_rank.csv")
    n_sig = int((rank["padj"] < 0.05).sum())
    return n_sig, int(len(rank))


def _perm_dir(run, method, idx):
    return run["outdir"] / "permutations" / method / f"perm_{idx}"


def _permute_slice_coords(adata, rng, method):
    """在单个 2D 切片内打乱坐标（保持表达/切片归属/标注不变）。

    用同一个置换同时打乱 x/y（及 SpaGCN 的像素/网格列）与 obsm['spatial']，
    即“坐标标签置换”，破坏切片内空间-表达关系，但不改变切片级表达分布。
    """
    n = adata.n_obs
    perm = rng.permutation(n)
    if method == "spagcn":
        for col in ("x", "y", "x_pixel", "y_pixel", "x_array", "y_array"):
            if col in adata.obs.columns:
                adata.obs[col] = adata.obs[col].to_numpy()[perm]
    if "spatial" in adata.obsm and adata.obsm["spatial"] is not None:
        adata.obsm["spatial"] = adata.obsm["spatial"][perm].copy()
    return adata


def _run_method_script(method, h5ad: Path, outdir: Path, sample: str, device: str):
    py = src.find_python()
    script = ROOT / "src/py_models" / f"run_{src.METHOD_SUBDIRS[method]}.py"
    src.log_message(f"运行 {method} 切片 {sample}")
    p = subprocess.run(
        [py, str(script), "--h5ad", str(h5ad), "--outdir", str(outdir),
         "--sample", sample, "--device", device],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if p.returncode != 0:
        raise RuntimeError(f"{method} 返回非零: {p.returncode}")


def _run_one_spark(run, idx, seed):
    src_dir = run["method_dirs"]["spark"]
    loc = pd.read_csv(src_dir / "location.csv", index_col=0)
    # 置换：打乱坐标行（barcode 与坐标的对应关系），表达/基因不变
    loc_perm = loc.sample(frac=1.0, random_state=seed)
    loc_perm.index = loc.index

    out = _perm_dir(run, "spark", idx)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("counts.mtx", "genes.csv", "barcodes.csv"):
        tgt = out / f
        if not tgt.exists():
            os.symlink(src_dir / f, tgt)
    loc_perm.to_csv(out / "location.csv", index_label="barcode")
    return _run_sparkx(out, run["sample"])


def _run_one_slice(run, method, idx, seed, device):
    import anndata as ad
    from src.py_models import merge_slices

    sub = src.METHOD_SUBDIRS[method]
    sample = run["sample"]
    out = _perm_dir(run, method, idx)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    slice_roots = sorted((run["outdir"] / sub / "slices").glob("S*"),
                         key=lambda p: int(p.name[1:]))
    if not slice_roots:
        raise FileNotFoundError(
            f"未找到切片目录: {run['outdir']}/{sub}/slices/S*（先跑 export_3d_slices）")

    for sr in slice_roots:
        i = int(sr.name[1:])
        sample_slice = f"{sample}_S{i}"
        src_h5ad = sr / sub / f"{sample_slice}_{sub}.h5ad"
        if not src_h5ad.exists():
            raise FileNotFoundError(f"缺少已导出的切片 h5ad: {src_h5ad}")
        a = ad.read_h5ad(src_h5ad)
        _permute_slice_coords(a, rng, method)
        dst_root = out / "slices" / f"S{i}"
        dst_dir = dst_root / sub
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_h5ad = dst_dir / f"{sample_slice}_{sub}.h5ad"
        a.write(dst_h5ad)
        del a
        _run_method_script(method, dst_h5ad, dst_root, sample_slice, device)

    merged = merge_slices.merge(out, method, sample)
    n_sig = int((merged["padj"] < 0.05).sum())
    return n_sig, int(len(merged))


def run_one(run, method, idx, seed, device="auto"):
    if method == "spark":
        n_sig, n_genes = _run_one_spark(run, idx, seed)
    elif method in ("spagcn", "spaseg"):
        n_sig, n_genes = _run_one_slice(run, method, idx, seed, device)
    else:
        raise NotImplementedError(f"未实现 method={method} 的坐标置换 FPR")

    fpr = n_sig / n_genes if n_genes else np.nan
    res = {"perm": int(idx), "seed": int(seed), "n_genes": n_genes,
           "n_sig": n_sig, "fpr": fpr}
    out = _perm_dir(run, method, idx)
    (out / "result.json").write_text(json.dumps(res), encoding="utf-8")
    src.log_message(f"perm {idx}: n_sig={n_sig}/{n_genes} FPR={fpr:.4f}")
    return res


def collect(run, method):
    base = run["outdir"] / "permutations" / method
    files = sorted(base.glob("perm_*/result.json"))
    if not files:
        raise FileNotFoundError(f"{base} 下没有 perm_*/result.json")
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    df = pd.DataFrame(rows)
    fprs = df["fpr"].to_numpy(np.float64)
    summary = {
        "method": method,
        "n_perms": int(len(df)),
        "mean_fpr": float(np.mean(fprs)),
        "sd_fpr": float(np.std(fprs)),
        "mean_n_sig": float(np.mean(df["n_sig"])),
        "n_genes": int(df["n_genes"].iloc[0]),
    }
    out = base / "permutation_fpr.csv"
    df.to_csv(out, index=False)
    (base / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    src.log_message(f"FPR 汇总: mean={summary['mean_fpr']:.4f} "
                    f"sd={summary['sd_fpr']:.4f} ({summary['n_perms']} 次置换) -> {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="坐标置换 FPR（金标准假阳性率）")
    ap.add_argument("--dataset", default=None, help="dataset key")
    ap.add_argument("--h5ad", default=None, help="输入 h5ad（覆盖 dataset）")
    ap.add_argument("--outdir", default=None, help="输出根目录")
    ap.add_argument("--sample", default=None, help="样本标签")
    ap.add_argument("--method", default="spark", choices=["spark", "spagcn", "spaseg"],
                    help="方法（spark/spagcn/spaseg）")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="spagcn/spaseg 训练设备（默认 auto）")
    ap.add_argument("--n-perms", type=int, default=10, help="顺序模式置换次数（默认 10）")
    ap.add_argument("--perm-idx", type=int, default=None,
                    help="只跑第 idx 次置换（HPC job array 用）")
    ap.add_argument("--seed", type=int, default=0, help="基础随机种子")
    ap.add_argument("--collect", action="store_true", help="汇总已有 perm_*.json")
    args = ap.parse_args()

    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad, outdir=args.outdir,
                          sample=args.sample, methods=[args.method], eval3d=True)
    src.log_header(f"坐标置换 FPR: {args.method} / {run['sample']}")

    if args.collect:
        collect(run, args.method)
        return 0

    if args.perm_idx is not None:
        run_one(run, args.method, args.perm_idx, args.seed + args.perm_idx,
                device=args.device)
    else:
        for i in range(args.n_perms):
            run_one(run, args.method, i, args.seed + i, device=args.device)
        collect(run, args.method)
    src.log_message("完成", section="完成")
    return 0


if __name__ == "__main__":
    main()
