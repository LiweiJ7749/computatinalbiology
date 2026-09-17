# -*- coding: utf-8 -*-
"""subsample_robustness.py —— spot 子采样稳健性（§6 第 4 项，对应协议 2.4）

无需重跑任何方法：固定各方法已检出的基因集，只对**评价层**做 spot 子采样，
检验"方法排名 / 基因排名"是否随采样而漂移。

做法（每个数据集）：
  1. 用 evaluation.load_expr_and_coords 读入统一表达/坐标（与正式评价完全同口径）；
  2. 各方法取 top-N 检出基因（N=--top-n，默认 500），在**全 spot** 上算 Moran's I，
     得到基线：方法间排名 + 每方法基因内部排名；
  3. 重复 B 次（--n-draws，默认 10）：随机抽 80% spot（--frac），重建 kNN 空间权重 W，
     对同一批基因重算 Moran，记录：
       - per-method：与基线基因排名的 Spearman ρ（基因级稳定性）；
       - 方法级：median Moran 的方法排名是否与基线一致（方法级稳定性）。
  4. 输出 ``subsample_robustness.csv``（长表）+ 打印摘要。

仅支持 2D 数据集（w_def=iso_2d）；3D 需保持切片归属，留待后续。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402
from src.utils import metrics as M  # noqa: E402
from src.utils import evaluation as EV  # noqa: E402

ALL_METHODS = list(src.ALL_METHODS)


def _method_ranking(medians: dict) -> dict:
    """按 median Moran 降序给方法排名（1=最好）。"""
    items = [(m, v) for m, v in medians.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1], reverse=True)
    return {m: i + 1 for i, (m, _) in enumerate(items)}


def _moran_for_genes(expr_mat, rows, W) -> np.ndarray:
    """对 (genes x spots) 稀疏矩阵的指定行子集算 Moran's I，返回数组。"""
    out = np.empty(len(rows), dtype=np.float64)
    for i, gr in enumerate(rows):
        x = M._dense_row(expr_mat, int(gr))
        out[i] = M.morans_i(x, W)
    return out


def run_dataset(dataset: str, top_n: int, n_draws: int, frac: float, seed: int,
                knn: int) -> list:
    run = src.resolve_run(dataset=dataset, eval3d=True)
    if int(run.get("dim") or 2) != 2:
        src.log_message(f"跳过 {dataset}（dim=3 暂不支持子采样稳健性）")
        return []

    expr_mat, W, gene_names, coords, *_ = EV.load_expr_and_coords(run, knn)
    n_spots = expr_mat.shape[1]
    gidx = {g: i for i, g in enumerate(gene_names)}

    # 各方法 top-N 基因（按 rank CSV），并映射到表达矩阵行
    method_genes, method_rows = {}, {}
    for m in ALL_METHODS:
        csv_path = run["method_dirs"][m] / \
            f"SVG_{EV.RANK_CSV_PREFIX[m]}_{run['sample']}_rank.csv"
        if not csv_path.exists():
            continue
        df = EV.read_rank_csv(csv_path)
        genes = [g for g in df["gene"].tolist() if g in gidx][:top_n]
        if len(genes) < 10:
            continue
        method_genes[m] = genes
        method_rows[m] = np.array([gidx[g] for g in genes], dtype=np.int64)
    if len(method_genes) < 2:
        src.log_message(f"跳过 {dataset}（可比方法不足 2）")
        return []

    # 基线（全 spot）
    full_moran = {m: _moran_for_genes(expr_mat, method_rows[m], W)
                  for m in method_genes}
    full_median = {m: float(np.nanmedian(v)) for m, v in full_moran.items()}
    base_rank = _method_ranking(full_median)

    rng = np.random.default_rng(seed)
    n_keep = max(10, int(round(frac * n_spots)))
    rows = []
    for b in range(n_draws):
        idx = np.sort(rng.choice(n_spots, size=n_keep, replace=False))
        sub_coords = coords[idx]
        W_sub, _ = M.spatial_weights(sub_coords, k=knn, w_def="iso_2d")
        sub_median = {}
        for m, grow in method_rows.items():
            # 只取子采样 spot 列，重算 Moran
            sub_mat = expr_mat[grow, :][:, idx]
            morans = _moran_for_genes(sub_mat, np.arange(len(grow)), W_sub)
            rho = M.spearman_rho(full_moran[m], morans)
            sub_median[m] = float(np.nanmedian(morans))
            rows.append({"dataset": dataset, "draw": b, "method": m,
                         "metric": "gene_rank_spearman", "value": rho,
                         "n_spots": n_keep, "n_genes": int(len(grow))})
        sub_rank = _method_ranking(sub_median)
        for m in sub_median:
            rows.append({"dataset": dataset, "draw": b, "method": m,
                         "metric": "median_moran", "value": sub_median[m],
                         "n_spots": n_keep, "n_genes": int(len(method_rows[m]))})
            consistent = int(sub_rank.get(m) == base_rank.get(m))
            rows.append({"dataset": dataset, "draw": b, "method": m,
                         "metric": "method_rank_consistent", "value": consistent,
                         "n_spots": n_keep, "n_genes": int(len(method_rows[m]))})

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="spot 子采样稳健性（协议 2.4）")
    ap.add_argument("--datasets",
                    default="mouse_brain_STARmap,DLPFC_151507,DLPFC_151510",
                    help="逗号分隔数据集 key（仅 2D）")
    ap.add_argument("--top-n", type=int, default=500, help="每方法取 top-N 基因")
    ap.add_argument("--n-draws", type=int, default=10, help="子采样重复次数")
    ap.add_argument("--frac", type=float, default=0.8, help="每次保留的 spot 比例")
    ap.add_argument("--knn", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "summary"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        src.log_message(f"子采样稳健性: {ds}")
        all_rows.extend(run_dataset(ds, args.top_n, args.n_draws, args.frac,
                                    args.seed, args.knn))
    if not all_rows:
        src.log_message("无可用结果")
        return 0

    df = pd.DataFrame(all_rows)
    df.to_csv(outdir / "subsample_robustness.csv", index=False)

    src.log_header("spot 子采样稳健性（80% spot × 10 次）")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"\n[{ds}]")
        for m in ALL_METHODS:
            s = sub[(sub["method"] == m) &
                    (sub["metric"] == "gene_rank_spearman")]
            c = sub[(sub["method"] == m) &
                    (sub["metric"] == "method_rank_consistent")]
            if s.empty:
                continue
            print(f"  {src.METHOD_LABELS[m]:8s} 基因排名 ρ={s['value'].median():.3f} "
                  f"方法排名一致率={c['value'].mean():.2f}")
    src.log_message(f"子采样稳健性已产出: {outdir / 'subsample_robustness.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())