# -*- coding: utf-8 -*-
"""merge_slices.py —— 把 2D 方法在 3D 数据上逐切片产出的 *_rank.csv 合并为单一排名。

合并口径（保守，docs/3d_svg_detection.md §5.3）：
  - 同一基因取“实际被测到的切片”未校正 p 值的中位数作为合并 p 值：
        p_g = median_i(p_i)，i ∈ {该基因被测到的切片}
    相比 Fisher（-2Σln p 会因相邻切片相关而高估显著性），中位数不假设切片独立、稳健。
  - 为避免“只在极少数切片被测到”的基因误导排名，默认要求至少 min_tested 片被测到才参与合并
    （否则视为无法跨切片整合，不进入排名）。
  - 合并后对保留基因做 BH 校正，输出列与统一排名 CSV 一致：gene,stat,pval,padj,rank。

用法（项目根，envs/spatial 的 python）：
  python src/py_models/merge_slices.py --method spagcn --method-dir results/local_results/zebrafish_3hpf/spaGCN --sample zebrafish_3hpf --min-tested 2
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402

# CSV 文件名前缀（与各方法脚本写出的 *_rank.csv 前缀一致）
_PREFIX = {"spark": "SPARK", "nnsvg": "nnSVG", "spagcn": "spaGCN", "spaseg": "spaSEG"}


def _find_slice_csvs(method_dir: Path, method: str, sample: str):
    prefix = _PREFIX[method]
    pat = f"SVG_{prefix}_{sample}_S*_rank.csv"
    hits = sorted(
        method_dir.rglob(pat),
        key=lambda p: int(re.search(r"_S(\d+)_rank\.csv$", p.name).group(1)),
    )
    if not hits:
        raise FileNotFoundError(f"{method_dir} 下未找到 {pat}")
    return hits


def _bh_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正（升序累计最小值实现）。"""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = p[order] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(n)
    out[order] = adj
    return out


def merge(method_dir: Path, method: str, sample: str, out: Path = None,
          min_tested: int = 2):
    paths = _find_slice_csvs(method_dir, method, sample)
    src.log_message(f"合并 {len(paths)} 个切片: {[p.name for p in paths]}")

    slice_maps = []
    all_genes = set()
    for p in paths:
        df = pd.read_csv(p, dtype={"gene": str})
        smap = {}
        for gene, pval, padj in zip(df["gene"], df.get("pval", [None] * len(df)),
                                    df.get("padj", [None] * len(df))):
            g = str(gene)
            p = pval if pd.notna(pval) else padj
            if pd.notna(p) and 0 < p <= 1:
                smap[g] = float(p)
        slice_maps.append(smap)
        all_genes.update(smap.keys())

    genes = sorted(all_genes)
    src.log_message(f"基因并集 = {len(genes)}")

    merged_p = {}
    tested_n = {}
    for g in genes:
        ps = [sm[g] for sm in slice_maps if g in sm]
        tested_n[g] = len(ps)
        # 仅对被测到的切片取中位数；不足 min_tested 片不进入排名
        merged_p[g] = float(np.median(ps)) if len(ps) >= min_tested else np.nan

    keep = [g for g in genes if tested_n[g] >= min_tested]
    dropped = len(genes) - len(keep)
    if dropped:
        src.log_message(f"少于 {min_tested} 片被测而排除的基因: {dropped} 个")

    g_arr = np.array(keep)
    p_arr = np.array([merged_p[g] for g in g_arr])
    padj = _bh_correct(p_arr)

    res = pd.DataFrame({
        "gene": g_arr,
        "stat": -np.log10(np.clip(p_arr, 1e-300, None)),
        "pval": p_arr,
        "padj": padj,
        "rank": 0,
    })
    res = res.sort_values(["padj", "pval", "gene"]).reset_index(drop=True)
    res["rank"] = res.index + 1

    out = out or (method_dir / f"SVG_{_PREFIX[method]}_{sample}_rank.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    src.log_message(f"已保存合并排名: {out} ({len(res)} 个基因)")

    tn = np.array([tested_n[g] for g in keep])
    if len(tn):
        src.log_message(f"每基因被测切片数分布: min={tn.min()} median={int(np.median(tn))} "
                        f"max={tn.max()}（切片总数 {len(paths)}）")
    return res


def main():
    ap = argparse.ArgumentParser(description="合并逐切片 rank CSV（p 值中位数 + BH）")
    ap.add_argument("--method", required=True, choices=list(_PREFIX),
                    help="方法名（nnsvg/spagcn/spaseg）")
    ap.add_argument("--method-dir", required=True, help="方法输出目录（如 .../nnSVG）")
    ap.add_argument("--sample", required=True, help="样本标签（不含 _S<i> 后缀）")
    ap.add_argument("--min-tested", type=int, default=2,
                    help="至少被测到的切片数（默认 2）")
    ap.add_argument("--out", default=None, help="合并输出 CSV（默认 <method-dir>/SVG_<METHOD>_<sample>_rank.csv）")
    args = ap.parse_args()

    src.log_header(f"合并切片排名: {args.method} / {args.sample}")
    merge(Path(args.method_dir), args.method, args.sample,
          Path(args.out) if args.out else None, min_tested=args.min_tested)
    src.log_message("合并完成", section="完成")


if __name__ == "__main__":
    main()
