#!/bin/bash
#SBATCH --job-name=bc_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o bc_cpu.%j.out
#SBATCH -e bc_cpu.%j.err

# Human_Breast_Cancer（16um h5ad，约 47 万 bin × 1.9 万基因）：全流程 CPU 单作业
# - 方法 spark,spagcn,spaseg（nnSVG 由 datasets.json 的 exclude_methods 排除）
# - 前置：先跑 Visium_HD_Human_Breast_Cancer_convert.sh 生成 16um h5ad
# - SPARK-X 全分辨率（投影核 single）；SpaGCN 按 bin_factor=8 聚合成 ~7k meta-spot
# - eval 末尾自动执行；X.toarray 约 71GB，配合 768G 节点可控（注意峰值内存）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
# 让单进程多线程方法（SpaSEG/SpaGCN 的 torch、scanpy/BLAS）用满分配的 CPU，避免退化为单核
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Human_Breast_Cancer \
  --methods spark,spagcn,spaseg \
  --device cpu
