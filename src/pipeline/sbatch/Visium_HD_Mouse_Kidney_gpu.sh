#!/bin/bash
#SBATCH --job-name=hd_kidney_spaseg
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=12:00:00
#SBATCH -o hd_kidney_spaseg.%j.out
#SBATCH -e hd_kidney_spaseg.%j.err

# Visium_HD_Mouse_Kidney：预处理 + SpaSEG（CPU）
# 说明：SpaSEG 训练 CNN 极小（GPU 仅用 1.5GB 显存/35s），后续 detect_svg 的
#      rank_genes_groups 是单核 CPU 算法，GPU 利用率过低，故直接放 CPU 运行。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Mouse_Kidney \
  --methods spaseg \
  --skip-eval \
  --device cpu
