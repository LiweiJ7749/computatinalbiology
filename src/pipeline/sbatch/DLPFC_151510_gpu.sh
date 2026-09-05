#!/bin/bash
#SBATCH --job-name=dlpfc510_gpu
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o dlpfc510_gpu.%j.out
#SBATCH -e dlpfc510_gpu.%j.err

# DLPFC_151510（4226 spot）：SpaGCN + SpaSEG（GPU，复用 cpu 作业的前处理）
# 资源预估：数千 spot，SpaGCN 邻接 4226² 可接受 → gpu_v100（V100 32G）。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset DLPFC_151510 \
  --methods spagcn,spaseg \
  --skip-preprocess --skip-eval \
  --device cuda
