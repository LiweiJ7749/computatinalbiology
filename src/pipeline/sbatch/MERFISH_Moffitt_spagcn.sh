#!/bin/bash
#SBATCH --job-name=merfish_spagcn
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=12:00:00
#SBATCH -o merfish_spagcn.%j.out
#SBATCH -e merfish_spagcn.%j.err

# MERFISH_Moffitt（103万 spot × 161 基因）—— 仅重跑 SpaGCN
# 背景：本机重跑 SpaGCN 因 56k meta-spot 建邻接矩阵（O(n^2)）超出 8GB 内存被 OOM，
#       改在 HPC 大内存节点（6126-24C-768G，768GB RAM）上重跑。
# - 前处理会按 run_params bin_factor=4 生成干净的 <sample>_spaGCN.h5ad（含 NaN 清洗，见 src/_raw_counts_matrix）
# - --device cpu：SpaGCN 建图/训练以内存为主，CPU 即可，不占 GPU 卡
# - 仅 methods=spagcn：SPARK-X / SpaSEG 已在本地用干净输入重跑完成
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
# 让单进程多线程方法（torch / scanpy / BLAS）用满分配的 CPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

bash src/pipeline/models_benchmark.sh \
  --dataset MERFISH_Moffitt \
  --methods spagcn \
  --device cpu
