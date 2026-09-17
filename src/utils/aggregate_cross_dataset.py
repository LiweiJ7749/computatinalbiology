# -*- coding: utf-8 -*-
"""aggregate_cross_dataset.py —— 跨数据集结果聚合器

把 ``results/local_results/**/eval/summary.json``（单数据集评价产物）聚合为
跨数据集主表，供"结果探究"第一步的数据盘点与后续平台分层对比使用。

产出（默认写到 ``results/summary/``，与 local_results 平级，不参与再次扫描）：
  1. datasets.csv        每个数据集一行：平台/维度/规模/实测规模/真值标注/方法数/kendall_w
  2. metrics_long.csv    长表：dataset × method × metric × value（含 random 基线）
  3. metrics_wide.csv    宽表：dataset 行 × (method__核心指标) 列
  4. method_coverage.csv 方法覆盖矩阵：dataset × 4 方法 0/1
  5. consistency.csv     两两方法一致性：spearman ρ 与 top-K Jaccard（上三角）
  6. consensus_genes.json 每个数据集的共识基因（top-100）

设计要点：
  * summary.json 存在新旧两版：新版含 ``dim/n_slices/w_def``，旧版（DLPFC/STARmap AD/
    Visium OB）没有，须回退 configs/datasets.json（dim、n_slices、tech、scale）。
  * ``dataset`` 字段是权威主键（与 configs key 一致；sample 可能不同，如 Slide_seq
    的 sample=Slide_seq_OB2、mouse_brain_STARmap 的 sample=STARmap_Mouse_Brain）。
  * 3D 数据集 SpaGCN/SpaSEG 走逐切片合并，wall_seconds 为 null，保持空值不填充。
  * 只 import 标准库 + pandas，不触发 anndata/scipy 等重依赖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src  # noqa: E402  (顶层仅标准库，import 开销低；取 ALL_METHODS/METHOD_LABELS)

ALL_METHODS = list(src.ALL_METHODS)          # ["spark", "nnsvg", "spagcn", "spaseg"]
RANDOM_KEY = "random"

# 平台族分层（tech -> 粗粒度平台类别），服务于后续"平台分层下结论"。
PLATFORM_GROUP = {
    "Visium": "array",
    "Visium_HD": "array",
    "DLPFC": "array",
    "STARmap": "imaging",
    "MERFISH": "imaging",
    "Slide_seq": "high_density",
    "Stereo_seq": "high_density",
    "Stereo_seq_zf": "high_density",
}

# 宽表里保留的核心指标（取值数值型、跨方法可比）。
CORE_METRICS = [
    "n_sig", "n_ranked",
    "median_moran_I", "median_geary_C_star", "moran_effect_z", "null_p",
    "rank_vs_moran_rho", "qq_lambda", "ks_uniform_p",
    "frac_low_spots", "median_n_spots", "median_mean_expr",
    "region_eta2", "region_eta2_mean",
    "ari_top100", "ari_top500", "ari_top1000",
    "nmi_top100", "nmi_top500", "nmi_top1000",
    "wall_seconds",
]


def _load_configs() -> dict:
    """读取 configs/datasets.json，返回 {dataset_key: entry}（原样）。"""
    path = ROOT / "configs" / "datasets.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_summaries(results_dir: Path):
    """递归查找 eval/summary.json，返回 [(dataset_key, Path)]。"""
    out = []
    for p in sorted(results_dir.rglob("eval/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        key = d.get("dataset") or p.parent.parent.name
        out.append((key, p))
    return out


def _find_fpr_summaries(results_dir: Path):
    """递归查找坐标置换 FPR 产物 summaries（permutations/<method>/summary.json）。

    返回 [(sample_dir_name, method, Path)]：
      - sample_dir_name = permutations 目录的上一级目录名（即 sample/dataset 目录名）；
      - method = summary.json 的 ``method`` 字段（缺失则用父目录名兜底）。
    """
    out = []
    for p in sorted(results_dir.rglob("permutations/*/summary.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        method = d.get("method") or p.parent.name
        sample_dir = p.parent.parent.parent.name
        out.append((sample_dir, method, p))
    return out


def _build_fpr_rows(results_dir: Path, configs: dict) -> list:
    """把坐标置换 FPR summary 汇总为长表行。

    返回字段：dataset / sample_dir / method / n_perms / mean_fpr / sd_fpr /
    mean_n_sig / n_genes。``dataset`` 常量 key 通过 configs 的 ``sample`` 字段
    反查（目录名 == sample 即命中）；查不到时回退目录名本身。
    """
    sample_to_dataset = {
        v.get("sample"): k for k, v in configs.items() if v.get("sample")
    }
    rows = []
    for sample_dir, method, path in _find_fpr_summaries(results_dir):
        d = json.loads(path.read_text(encoding="utf-8"))
        dataset = sample_to_dataset.get(sample_dir, sample_dir)
        rows.append({
            "dataset": dataset,
            "sample_dir": sample_dir,
            "method": method,
            "n_perms": d.get("n_perms"),
            "mean_fpr": d.get("mean_fpr"),
            "sd_fpr": d.get("sd_fpr"),
            "mean_n_sig": d.get("mean_n_sig"),
            "n_genes": d.get("n_genes"),
        })
    return rows


def _flatten_method_metrics(m: dict) -> dict:
    """把一个方法的指标 dict 展平成标量 dict；info_flags -> info_has_pval 等前缀。"""
    flat = {}
    for k, v in m.items():
        if isinstance(v, dict):                       # info_flags
            for fk, fv in v.items():
                flat[f"info_{fk}"] = fv
        elif isinstance(v, (list, tuple)):
            continue                                   # 共识基因等非标量不进入长表
        else:
            flat[k] = v
    return flat


def _build_rows(summaries, configs):
    """遍历 summary.json，产出 (datasets_rows, metrics_rows, coverage_rows,
    consistency_rows, consensus_map)。"""
    dataset_rows = []
    metric_rows = []
    coverage_rows = []
    consistency_rows = []
    consensus_map = {}

    for key, path in summaries:
        d = json.loads(path.read_text(encoding="utf-8"))
        cfg = configs.get(key, {}) or {}

        tech = cfg.get("tech") or ""
        scale = cfg.get("scale") or ""
        platform_group = PLATFORM_GROUP.get(tech, "")

        # dim/n_slices/w_def：新版 summary 有，旧版回退 configs，再兜底。
        dim = d.get("dim")
        if dim is None:
            dim = int(cfg.get("dim") or 2)
        dim = int(dim)
        n_slices = d.get("n_slices")
        if n_slices is None:
            n_slices = int(cfg.get("n_slices") or (1 if dim == 2 else 0))
        n_slices = int(n_slices)
        w_def = d.get("w_def") or (f"iso_{dim}d" if dim in (2, 3) else None)

        has_labels = bool(d.get("has_labels"))
        label_col = d.get("label_col")
        methods = d.get("methods", {})
        method_list = [m for m in ALL_METHODS if m in methods]
        random_base = d.get("random_baseline") or {}

        # --- 1) 数据集级一行 ---
        dataset_rows.append({
            "dataset": key,
            "sample": d.get("sample"),
            "tech": tech,
            "platform_group": platform_group,
            "dim": dim,
            "scale": scale,
            "n_spots": d.get("n_spots"),
            "n_genes": d.get("n_genes"),
            "n_slices": n_slices,
            "w_def": w_def,
            "has_labels": has_labels,
            "label_col": label_col,
            "n_methods": len(method_list),
            "methods": ",".join(method_list),
            "kendall_w": d.get("consistency", {}).get("kendall_w"),
        })

        # --- 2) 方法覆盖矩阵（4 方法 0/1）---
        cov = {"dataset": key}
        for m in ALL_METHODS:
            cov[m] = 1 if m in methods else 0
        coverage_rows.append(cov)

        # --- 3) 长表：真实方法 + random 基线 ---
        for m in ALL_METHODS:
            if m not in methods:
                continue
            flat = _flatten_method_metrics(methods[m])
            for metric, value in flat.items():
                metric_rows.append({
                    "dataset": key, "tech": tech, "platform_group": platform_group,
                    "dim": dim, "scale": scale, "method": m,
                    "metric": metric, "value": value,
                })
        for metric, value in _flatten_method_metrics(random_base).items():
            metric_rows.append({
                "dataset": key, "tech": tech, "platform_group": platform_group,
                "dim": dim, "scale": scale, "method": RANDOM_KEY,
                "metric": metric, "value": value,
            })

        # --- 4) 一致性（上三角）---
        cons = d.get("consistency", {}) or {}
        spearman = cons.get("spearman_matrix") or {}
        jaccard = cons.get("jaccard_matrix") or {}
        for i, a in enumerate(method_list):
            for b in method_list[i + 1:]:
                consistency_rows.append({
                    "dataset": key, "tech": tech, "platform_group": platform_group,
                    "dim": dim, "method_a": a, "method_b": b,
                    "spearman": (spearman.get(a) or {}).get(b),
                    "jaccard": (jaccard.get(a) or {}).get(b),
                })

        # --- 5) 共识基因（键名固定为 top100）---
        cons_genes = cons.get("consensus_genes_top100")
        if cons_genes:
            consensus_map[key] = cons_genes

    return (dataset_rows, metric_rows, coverage_rows, consistency_rows, consensus_map)


def _build_wide(metric_rows) -> pd.DataFrame:
    """从长表抽核心指标，转成宽表：行=dataset，列=method__metric。"""
    df = pd.DataFrame(metric_rows)
    if df.empty:
        return df
    core = df[df["metric"].isin(CORE_METRICS)].copy()
    core["col"] = core["method"] + "__" + core["metric"]
    wide = core.pivot_table(index="dataset", columns="col", values="value",
                            aggfunc="first")
    # 列按固定方法顺序 x 核心指标顺序排列
    ordered_cols = [f"{m}__{mm}" for m in ALL_METHODS + [RANDOM_KEY]
                    for mm in CORE_METRICS if f"{m}__{mm}" in wide.columns]
    wide = wide.reindex(columns=ordered_cols).reset_index()   # dataset 恢复为列
    return wide


def _write(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    path = outdir / name
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="跨数据集聚合 eval/summary.json")
    ap.add_argument("--results-dir", default=str(ROOT / "results" / "local_results"),
                    help="扫描根目录（默认 results/local_results）")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "summary"),
                    help="输出目录（默认 results/summary）")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    configs = _load_configs()
    summaries = _find_summaries(results_dir)
    src.log_message(f"找到 {len(summaries)} 个 eval/summary.json")

    dataset_rows, metric_rows, coverage_rows, consistency_rows, consensus_map = \
        _build_rows(summaries, configs)

    datasets_df = pd.DataFrame(dataset_rows)
    metrics_long = pd.DataFrame(metric_rows)
    metrics_wide = _build_wide(metric_rows)
    coverage_df = pd.DataFrame(coverage_rows)
    consistency_df = pd.DataFrame(consistency_rows)
    fpr_df = pd.DataFrame(_build_fpr_rows(results_dir, configs))

    paths = [
        _write(datasets_df, outdir, "datasets.csv"),
        _write(metrics_long, outdir, "metrics_long.csv"),
        _write(metrics_wide, outdir, "metrics_wide.csv"),
        _write(coverage_df, outdir, "method_coverage.csv"),
        _write(consistency_df, outdir, "consistency.csv"),
        _write(fpr_df, outdir, "fpr.csv"),
    ]
    with (outdir / "consensus_genes.json").open("w", encoding="utf-8") as f:
        json.dump(consensus_map, f, ensure_ascii=False, indent=2)

    src.log_header("聚合完成")
    src.log_message(f"数据集: {len(datasets_df)}；长表行数: {len(metrics_long)}；"
                    f"方法覆盖: {len(coverage_df)}；一致性对上三角: {len(consistency_df)}")
    for p in paths:
        src.log_message(f"  {p.relative_to(ROOT)}")
    src.log_message(f"  {outdir.relative_to(ROOT)}/consensus_genes.json")

    # 方法覆盖矩阵快速自检（打印到 stdout）
    print("\n方法覆盖矩阵（方法数 × 数据集数）:")
    print(coverage_df[ALL_METHODS].sum().rename(index=src.METHOD_LABELS).to_string())
    print("\n按 platform_group 统计数据集数:")
    print(datasets_df["platform_group"].value_counts(dropna=False).to_string())
    if not fpr_df.empty:
        print("\n坐标置换 FPR 汇总:")
        print(fpr_df[["dataset", "method", "n_perms", "mean_fpr", "sd_fpr",
                      "mean_n_sig", "n_genes"]].to_string(index=False))
    else:
        print("\n坐标置换 FPR 汇总: 无 permutations/*/summary.json（未跑 FPR）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
