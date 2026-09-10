#!/bin/bash
#SBATCH --job-name=svg3d_export
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=2:00:00
#SBATCH -o svg3d_export.%j.out
#SBATCH -e svg3d_export.%j.err
#
# 3D 数据逐切片导出（SpaGCN/SpaSEG 输入）。
# 提交（在项目根 ~/svg_methods）：
#   sbatch --export=ALL,DATASET=Stereo_seq_drosophila src/pipeline/sbatch/run_3d_slices_export_cpu.sh
set -euo pipefail

DATASET="${DATASET:-Stereo_seq_drosophila}"

cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset "$DATASET" \
  --methods spagcn,spaseg \
  --skip-slices --skip-merge --skip-eval
