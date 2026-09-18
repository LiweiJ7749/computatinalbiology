#!/bin/bash
#SBATCH --job-name=merfish_full
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o merfish_full.%j.out
#SBATCH -e merfish_full.%j.err

# ============================================================
# MERFISH_Moffitt（103万 spot × 161 基因）—— 一键全流程重跑 + eval
# ============================================================
# 用途：在 HPC 大内存节点上，用【干净输入】重跑 MERFISH 全部方法，
#       作业末尾自动执行 evaluation.py，直接产出 eval 结果与图。
#
# 说明：
#  1) 方法 = spark, spagcn, spaseg（nnSVG 由 datasets.json 的
#     exclude_methods 排除——历史上因 1M spot 下 BRISC 2D 运行时间过长而舍弃）。
#     若确要强制加入 nnSVG，见文件末尾"可选：强制启用 nnSVG"。
#  2) 前处理按 run_params bin_factor=4 聚合 SpaGCN/nnSVG 输入；
#     本仓库代码已含 NaN 清洗（src/_raw_counts_matrix），
#     counts.mtx 与 *_spaGCN/*_spaSEG.h5ad 均为干净计数。
#  3) --device cpu：SpaGCN 内存瓶颈（768GB 节点足够）、SPARK-X/SpaSEG 无需 GPU。
#  4) 提交前请先把本地已重跑好的 SPARK_X/、spaSEG/ 结果同步到 HPC
#     （results/local_results/MERFISH_Moffitt/），保持三方法一致。
# ============================================================
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
# 让单进程多线程方法（torch / scanpy / BLAS）用满分配的 CPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

# 不指定 --methods = 运行该数据集全部可用方法（自动排除 nnSVG），
# 末尾自动执行 evaluation.py（eval/summary.json + 全部图 + 表）
bash src/pipeline/models_benchmark.sh \
  --dataset MERFISH_Moffitt \
  --device cpu

# ============================================================
# 可选：强制启用 nnSVG（第 4 种方法）——
#   1) 提交前临时把 configs/datasets.json 里 MERFISH_Moffitt 的
#      "exclude_methods": ["nnsvg"] 改为 []（或删除该键）；
#   2) 上面命令改为：--methods spark,nnsvg,spagcn,spaseg；
#   3) 跑完后再把 exclude_methods 改回 ["nnsvg"]。
#   注意：nnSVG 在 56k meta-spot × 161 基因上估算 10–60 分钟量级，
#         若超时/失败不影响其他方法（作业按顺序执行，eval 只汇总已有结果）。
# ============================================================
