#!/bin/bash
#SBATCH --job-name=hd_kidney_cpu
#SBATCH --partition=7542-64C-512G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=24:00:00
#SBATCH -o hd_kidney_cpu.%j.out
#SBATCH -e hd_kidney_cpu.%j.err

# Visium_HD_Mouse_Kidney（502009 spot × 19336 基因）：预处理 + SPARK-X + SpaGCN + SpaSEG
# - SPARK-X 全分辨率
# - SpaGCN 用 bin_factor=2（126113 meta-spot，邻接矩阵约 63.6GB，峰值 ~190GB）
# - SpaSEG 放 CPU（训练 CNN 极小、detect_svg 单核，GPU 利用率低，放 CPU 省 GPU 卡）
# - 放弃 nnSVG（BRISC 逐基因太慢，7.5h 未完成）
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/models_benchmark.sh \
  --dataset Visium_HD_Mouse_Kidney \
  --methods spark,spagcn,spaseg \
  --device cpu \
  --skip-eval
