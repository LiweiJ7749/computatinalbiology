#!/bin/bash
#SBATCH --job-name=bc_pre
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=8:00:00
#SBATCH -o bc_pre.%j.out
#SBATCH -e bc_pre.%j.err

# Human_Breast_Cancer 前处理：读转换后的大 h5ad，生成 SPARK-X 输入。
# 说明：h5ad 为 Visium_HD_Kidney 的数倍，仅 SPARK-X 线性可扩展，故只生成 spark。
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

$SVG_PYTHON src/preprocess/h5ad_preprocess.py \
  --dataset Visium_HD_Human_Breast_Cancer \
  --methods spark
