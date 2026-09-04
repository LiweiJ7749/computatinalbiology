#!/bin/bash
#SBATCH --job-name=dlpfc508_gpu
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o dlpfc508_gpu.%j.out
#SBATCH -e dlpfc508_gpu.%j.err

# DLPFC_151508：SpaGCN + SpaSEG（GPU，复用 cpu 作业已生成的前处理）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset DLPFC_151508 \
  --methods spagcn,spaseg \
  --skip-preprocess --skip-eval \
  --device cuda
