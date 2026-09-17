# -*- coding: utf-8 -*-
"""marker_enrichment.py —— 已知 marker 回收 / 富集（§6 第 2 项，通用版）

对多数据集用**文献公认、独立于本项目**的 marker 列表评估各方法：
  1) top-K marker 回收率 Precision@K；
  2) 显著集合富集倍数 FE = (k/K)/(M/N) + Fisher 精确检验（metrics.marker_enrichment）。

背景基因数 N 取各方法自身的 rank 基因数。支持两类 marker 集合：
  - DLPFC_*     : 人脑皮层 laminar marker（spatialLIBD / Maynard 2018 口径，大写）；
  - zebrafish_* : 斑马鱼胚胎发育核心空间 marker（ZFIN 口径，小写基因名）。
  - STARmap_AD 暂不支持：Zeng2023 AD 数据的 var.index 为纯数字 ID、无 gene symbol
    列，无法与基因符号 marker 匹配（脚本直接跳过并提示）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402
from src.utils import metrics as M  # noqa: E402

ALL_METHODS = list(src.ALL_METHODS)
RANK_CSV_PREFIX = {"spark": "SPARK", "nnsvg": "nnSVG", "spagcn": "spaGCN",
                   "spaseg": "spaSEG"}
TOP_K_LIST = (100, 500)

# 文献公认 marker（独立于本项目数据）
DLPFC_MARKERS = [
    "MBP", "MOBP", "PLP1", "CNP", "MAG",                       # WM
    "RELN", "CPLX2", "HTR1F",                                  # L1
    "CUX2", "LAMP5", "CARTPT", "PVALB",                        # L2/3
    "RORB", "PCP4", "WFS1", "ETV1",                            # L4
    "BCL11B", "FEZF2", "CRYM",                                 # L5
    "NR4A2", "SYNPR", "TLE4", "SERPINE2",                      # L6
]
ZEBRAFISH_MARKERS = [
    "tbxta", "tbx16", "sox32", "sox17", "gsc", "chrd", "shha", "krt8",
    "pax2a", "myod1", "egr2b", "sox10",
]


def _markers_for_dataset(dataset: str):
    if dataset.startswith("DLPFC"):
        return sorted(set(DLPFC_MARKERS))
    if dataset.startswith("zebrafish"):
        return sorted(set(ZEBRAFISH_MARKERS))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="已知 marker 回收/富集（DLPFC / 斑马鱼）")
    ap.add_argument("--datasets", default="DLPFC_151507,zebrafish_5hpf",
                    help="逗号分隔数据集 key")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "summary"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    rows = []
    for ds in datasets:
        markers = _markers_for_dataset(ds)
        if markers is None:
            src.log_message(f"跳过 {ds}（无 marker 集合 / STARmap AD 数字 ID 不支持）")
            continue
        base = ROOT / "results" / "local_results" / ds
        for m in ALL_METHODS:
            csv_path = base / src.METHOD_SUBDIRS[m] / \
                f"SVG_{RANK_CSV_PREFIX[m]}_{ds}_rank.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path, dtype={"gene": str})
            genes = df["gene"].tolist()
            n_bg = len(genes)
            valid = [g for g in markers if g in set(genes)]

            for k in TOP_K_LIST:
                top = set(df["gene"].head(k))
                hit = len(top & set(valid))
                rows.append({"dataset": ds, "method": m, "metric": f"precision@{k}",
                             "value": hit / k, "n_markers": len(valid),
                             "n_hit": hit, "n_bg": n_bg})

            sig = df.loc[df["padj"] < 0.05, "gene"].tolist()
            enr = M.marker_enrichment(sig, valid, n_bg)
            rows.append({"dataset": ds, "method": m, "metric": "enrichment_FE",
                         "value": enr["FE"], "n_markers": len(valid),
                         "n_hit": enr["k"], "n_bg": n_bg})
            rows.append({"dataset": ds, "method": m, "metric": "enrichment_fisher_p",
                         "value": enr["fisher_p"], "n_markers": len(valid),
                         "n_hit": enr["k"], "n_bg": n_bg})

    if not rows:
        src.log_message("无可用结果")
        return 0
    res = pd.DataFrame(rows)
    res.to_csv(outdir / "marker_enrichment.csv", index=False)

    src.log_header("已知 marker 富集（文献独立）")
    for ds in datasets:
        sub = res[res["dataset"] == ds]
        if sub.empty:
            continue
        for m in ALL_METHODS:
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            p100 = float(r[r["metric"] == "precision@100"]["value"].iloc[0])
            fe = float(r[r["metric"] == "enrichment_FE"]["value"].iloc[0])
            fp = float(r[r["metric"] == "enrichment_fisher_p"]["value"].iloc[0])
            print(f"[{ds}] {src.METHOD_LABELS[m]:8s} P@100={p100:.3f} "
                  f"FE={fe:.2f} fisher_p={fp:.1e}")
    src.log_message(f"marker 富集已产出: {outdir / 'marker_enrichment.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())