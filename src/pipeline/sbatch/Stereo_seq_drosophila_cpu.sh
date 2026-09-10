#!/bin/bash
#SBATCH --job-name=stereo3d_cpu
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=24:00:00
#SBATCH -o stereo3d_cpu.%j.out
#SBATCH -e stereo3d_cpu.%j.err

# Stereo_seq_drosophila（3D，15295 spot × 13668 基因 dense，1.7GB）：全流程 CPU 单作业
# 与 MERFISH 同风格，一个 sbatch 跑完：
#   1) SPARK-X 原生 3D（dim=3 仅 SPARK-X；--skip-eval 避免在此单独评估）
#   2) SpaGCN/SpaSEG 逐切片 2D + 跨切片合并
#   3) 三方法（SPARK-X + 合并后 SpaGCN/SpaSEG）评估 + 3D 空间可视化
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
# 让单进程多线程方法（SpaSEG/SpaGCN 的 torch、scanpy/BLAS）用满分配的 CPU，避免退化为单核
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

# 1) SPARK-X 原生 3D
bash src/pipeline/models_benchmark.sh \
  --dataset Stereo_seq_drosophila \
  --methods spark \
  --skip-eval

# 2) SpaGCN/SpaSEG 逐切片 + 合并 + 三方法评估 + 空间可视化
bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset Stereo_seq_drosophila \
  --methods spagcn,spaseg \
  --device cpu
