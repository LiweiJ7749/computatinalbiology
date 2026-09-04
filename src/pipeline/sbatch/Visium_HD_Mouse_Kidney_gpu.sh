#!/bin/bash
#SBATCH --job-name=hd_kidney_gpu
#SBATCH --partition=gpuB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH -o hd_kidney_gpu.%j.out
#SBATCH -e hd_kidney_gpu.%j.err

# Visium_HD_Mouse_Kidney：SpaGCN + SpaSEG（gpuB 80G 显存，复用 cpu 作业的前处理）
# 注意：50 万 spot 的邻接矩阵/网格可能内存显存吃紧，若 OOM 需降规模或换思路。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Mouse_Kidney \
  --methods spagcn,spaseg \
  --skip-preprocess --skip-eval \
  --device cuda
