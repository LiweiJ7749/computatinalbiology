#!/bin/bash
#SBATCH --job-name=zf10_cpu
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=24:00:00
#SBATCH -o zf10_cpu.%j.out
#SBATCH -e zf10_cpu.%j.err

# zebrafish_10hpf（Stereo_seq_zf，3D，19102 spot × 18698 基因，26 片）：全流程 CPU 单作业
# 与 MERFISH 同风格，一个 sbatch 跑完：
#   1) SPARK-X 原生 3D（坐标 obs spatial_x/y，z=slice）
#   2) SpaGCN/SpaSEG 逐切片 2D + 跨切片合并
#   3) 三方法评估 + 3D 空间可视化
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-24}

# 1) SPARK-X 原生 3D
bash src/pipeline/models_benchmark.sh \
  --dataset zebrafish_10hpf \
  --methods spark \
  --skip-eval

# 2) SpaGCN/SpaSEG 逐切片 + 合并 + 三方法评估 + 空间可视化
bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset zebrafish_10hpf \
  --methods spagcn,spaseg \
  --device cpu
