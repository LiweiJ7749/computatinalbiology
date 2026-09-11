#!/bin/bash
#SBATCH --job-name=slide3d_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=30:00:00
#SBATCH -o slide3d_cpu.%j.out
#SBATCH -e slide3d_cpu.%j.err

# Slide_seq_OB2_3D（3D，20 切片 × 约 88 万 spot，float64 约 4.5GB）：全流程 CPU 单作业
# 与 MERFISH 同风格，一个 sbatch 跑完：
#   1) SPARK-X 原生 3D（2D→3D 简单堆叠，dim=3 仅 SPARK-X）
#   2) SpaGCN/SpaSEG 逐切片 2D + 跨切片合并
#   3) 三方法评估 + 3D 空间可视化
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}

# 1) SPARK-X 原生 3D
bash src/pipeline/models_benchmark.sh \
  --dataset Slide_seq_OB2_3D \
  --methods spark \
  --skip-eval

# 2) SpaGCN/SpaSEG 逐切片 + 合并 + 三方法评估 + 空间可视化
bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset Slide_seq_OB2_3D \
  --methods spagcn,spaseg \
  --device cpu
