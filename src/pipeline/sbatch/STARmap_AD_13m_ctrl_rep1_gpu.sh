#!/bin/bash
#SBATCH --job-name=starad_gpu
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH -o starad_gpu.%j.out
#SBATCH -e starad_gpu.%j.err

# STARmap_AD_13m_ctrl_rep1（8034 spot）：SpaGCN + SpaSEG（GPU，复用 cpu 前处理）
# 资源预估：8034 spot，SpaGCN 邻接 8034² 可接受 → gpu_v100（V100 32G）。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset STARmap_AD_13m_ctrl_rep1 \
  --methods spagcn,spaseg \
  --skip-preprocess --skip-eval \
  --device cuda
