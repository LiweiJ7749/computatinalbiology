#!/bin/bash
#SBATCH --job-name=svg3d_slice
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --array=0-31
#SBATCH --time=6:00:00
#SBATCH -o svg3d_slice_%a.%j.out
#SBATCH -e svg3d_slice_%a.%j.err
#
# 逐切片运行 2D 方法（SpaGCN/SpaSEG）的 job array 模板。
#
# 提交（在项目根；先决条件：已运行 export_3d_slices.py，本模板 --skip-export）：
#   sbatch --export=ALL,DATASET=Stereo_seq_drosophila,N_SLICES=16 \
#          src/pipeline/sbatch/run_3d_slices_array_cpu.sh
#
# 典型 3D 完整流程：
#   1) SPARK-X 原生 3D：      sbatch src/pipeline/sbatch/Stereo_seq_drosophila_cpu.sh
#   2) 导出切片：              bash   src/pipeline/run_3d_slices_benchmark.sh --dataset <D> --skip-slices --skip-merge --skip-eval
#   3) 本 array（逐切片）：    sbatch --export=ALL,DATASET=<D>,N_SLICES=<n> 本脚本
#   4) 合并+评估+可视化：      sbatch --export=ALL,DATASET=<D> src/pipeline/sbatch/run_3d_slices_merge_cpu.sh
set -euo pipefail

DATASET="${DATASET:-Stereo_seq_drosophila}"
N_SLICES="${N_SLICES:-16}"

# array 上限 0-31：超过范围的数组任务直接退出（提交时按实际切片数设置 N_SLICES）
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$N_SLICES" ]; then
  echo "skip (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0} >= N_SLICES=$N_SLICES)"
  exit 0
fi

cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript

bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset "$DATASET" \
  --methods spagcn,spaseg \
  --slice-idx "$SLURM_ARRAY_TASK_ID" \
  --skip-export --skip-merge --skip-eval
