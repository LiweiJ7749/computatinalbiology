#!/bin/bash
#SBATCH --job-name=fpr_perm
#SBATCH --partition=6240-36C-192G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --array=0-19
#SBATCH --time=24:00:00
#SBATCH -o fpr_perm_%a.%j.out
#SBATCH -e fpr_perm_%a.%j.err
#
# 坐标置换 FPR（SPARK-X/SpaGCN/SpaSEG）：每个数组任务跑一次置换。
# 提交（在项目根 ~/svg_methods）：
#   sbatch --export=ALL,DATASET=Stereo_seq_drosophila,METHOD=spagcn,SEED=0,N_PERMS=20,DEVICE=cpu \
#          src/pipeline/sbatch/permutation_fpr_cpu.sh
# 完成后汇总：
#   sbatch --dependency=afterok:<array_job_id> --export=ALL,DATASET=Stereo_seq_drosophila,METHOD=spagcn \
#          src/pipeline/sbatch/permutation_fpr_collect_cpu.sh
set -euo pipefail

DATASET="${DATASET:-Stereo_seq_drosophila}"
METHOD="${METHOD:-spark}"
SEED="${SEED:-0}"
N_PERMS="${N_PERMS:-20}"
DEVICE="${DEVICE:-cpu}"

# 超出 N_PERMS 的数组任务直接退出（--array 上限写 0-31 时保证可复用）
if [ "${SLURM_ARRAY_TASK_ID:-0}" -ge "$N_PERMS" ]; then
  echo "skip (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0} >= N_PERMS=$N_PERMS)"
  exit 0
fi

cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}

"$SVG_PYTHON" src/utils/permutation_fpr.py \
  --dataset "$DATASET" \
  --method "$METHOD" \
  --device "$DEVICE" \
  --perm-idx "$SLURM_ARRAY_TASK_ID" \
  --seed "$SEED"
