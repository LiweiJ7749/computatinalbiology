#!/bin/bash
#SBATCH --job-name=bc_convert
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --time=6:00:00
#SBATCH -o bc_convert.%j.out
#SBATCH -e bc_convert.%j.err

# Human_Breast_Cancer：feature_slice.h5(1.6GB) -> h5ad（一次性数据准备，独立作业）
# - --bin-size 16：16um 聚合（368184 bin），规模约为 8um 的 1/4，控制 h5ad 体积
# - --n-jobs 36：多进程逐基因并行，榨取 6240-36C-192G 的全部 36 核
# - 内存实测：总 UMI 6.06 亿，16um 下累积峰值约 10~20GB；192G 有 ~13 倍余量，足够且不浪费稀缺 768G 节点
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python

$SVG_PYTHON src/preprocess/10xVisium_pretreat_h5toh5ad.py \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_feature_slice.h5 \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_11mm_Human_Breast_Cancer_spatial.tar.gz \
  data/10xVisium/Human_Breast_Cancer/Visium_HD_Human_Breast_Cancer.h5ad \
  --bin-size 16 --n-jobs 36
