# -*- coding: utf-8 -*-
"""h5ad_preprocess.py —— 统一空间转录组数据预处理（批量化薄 CLI）

四方法的"共同前处理"逻辑已全部上移到 ``src/__init__.py``（src.preprocess_run /
src.resolve_run / src.ensure_run_dirs），本脚本仅保留命令行入口，供手工调用与
pipeline（src/pipeline/models_benchmark.sh）使用：

  1. 解析 run 配置：--dataset（命中 src.DATASETS 注册表）或显式 --h5ad/--spatial/
     --outdir/--sample-name；--methods 控制要生成哪些方法的数据；
  2. 目录检查/建立（各方法子目录）；
  3. 一次生成四个方法可直接使用的数据：
       SPARK-X / nnSVG : counts.mtx(genes x spots) + genes.csv + barcodes.csv + location.csv
       SpaGCN          : <sample>_spaGCN.h5ad（obs 坐标列 + 组织学图像，可选）
       SpaSEG          : <sample>_spaSEG.h5ad（obsm["spatial"]）

用法（在项目根目录下，用 env_spatial 的 python）::

    # 数据集注册表里的 key（默认 mouse_brain_STARmap）
    python src/preprocess/h5ad_preprocess.py --dataset mouse_brain_STARmap
    python src/preprocess/h5ad_preprocess.py --dataset Visium_Mouse_Olfactory_Bulb

    # 自定义数据（dataset 为 None 时默认用 Visium 注册表便于演示）
    python src/preprocess/h5ad_preprocess.py \\
        --h5ad ./data/STARmap/mouse_brain_cortex/mouse_brain_STARmap_processed.h5ad \\
        --outdir ./results/local_results/mouse_brain_STARmap \\
        --sample-name STARmap_Mouse_Brain --methods spark nnsvg

    # 只生成 R 方法（SPARK-X / nnSVG）需要的数据
    python src/preprocess/h5ad_preprocess.py --dataset mouse_brain_STARmap --methods spark nnsvg
"""
import argparse
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 项目根（本文件位于 src/preprocess/ 下，parents[2] = 项目根）
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import src  # noqa: E402  （共同前处理/配置均在本模块）


def main() -> None:
    ap = argparse.ArgumentParser(
        description="统一 h5ad 预处理：生成 SPARK-X/nnSVG 数据 + SpaGCN/SpaSEG 接口",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--dataset", default=None,
                    help=f"数据集 key（见 src.DATASETS: {list(src.DATASETS)}）")
    ap.add_argument("--h5ad", type=str, default=None, help="输入 h5ad（绝对或相对项目根）")
    ap.add_argument("--spatial", type=str, default=None,
                    help="tissue_positions.csv 或 spatial.tar.gz（默认自动查找）")
    ap.add_argument("--outdir", type=str, default=None, help="输出根目录")
    ap.add_argument("--sample-name", type=str, default=None,
                    help="样本名（用于 SpaGCN/SpaSEG 输出文件前缀，默认取 dataset/h5ad）")
    ap.add_argument("--tech", type=str, default=None,
                    help=f"技术类型（用于区分 h5ad 结构差异）：{list(src.TECH_TYPES)}"
                         "；默认取 dataset 注册的 tech，未知则自动探测")
    ap.add_argument("--methods", nargs="+", choices=list(src.METHOD_SUBDIRS),
                    default=list(src.ALL_METHODS),
                    help="要生成的数据")
    args = ap.parse_args()

    # 完全未指定数据时，用默认演示数据集（Visium）
    if args.dataset is None and args.h5ad is None:
        args.dataset = "Visium_Mouse_Olfactory_Bulb"

    run = src.resolve_run(dataset=args.dataset, h5ad=args.h5ad,
                          spatial=args.spatial, outdir=args.outdir,
                          sample=args.sample_name, tech=args.tech,
                          methods=args.methods)
    src.log_header("h5ad_preprocess (batch)")
    src.log_message(f"dataset = {run['dataset']}")
    src.log_message(f"h5ad    = {run['h5ad']}")
    src.log_message(f"outdir  = {run['outdir']}")
    src.log_message(f"sample  = {run['sample']}")
    src.log_message(f"tech    = {run['tech'] or 'auto'}")
    src.log_message(f"methods = {run['methods']}")
    src.preprocess_run(run, methods=run["methods"])


if __name__ == "__main__":
    main()
