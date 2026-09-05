#!/bin/bash
#SBATCH --job-name=merfish_gpu
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o merfish_gpu.%j.out
#SBATCH -e merfish_gpu.%j.err

# MERFISH_Moffitt：SpaSEG（gpuB 80G；SpaGCN 因 dense 邻接 O(n^2) 对 103万 spot 不适用，已跳过）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset MERFISH_Moffitt \
  --methods spaseg \
  --skip-preprocess --skip-eval \
  --device cuda
