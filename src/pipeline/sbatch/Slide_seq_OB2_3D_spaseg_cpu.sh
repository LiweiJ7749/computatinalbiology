#!/bin/bash
#SBATCH --job-name=slide3d_spaseg
#SBATCH --partition=6126-24C-768G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --time=30:00:00
#SBATCH -o slide3d_spaseg.%j.out
#SBATCH -e slide3d_spaseg.%j.err

# Slide_seq_OB2_3D 续跑作业：只补 SpaSEG（上一条作业在 SpaSEG S0 因 KeyError: 'index' 中断）。
#
# 复用已有的 SPARK-X（原生 3D）与 SpaGCN（20 片 + 合并）结果，避免重算约 9 小时：
#   --methods spaseg : 只跑 SpaSEG 的逐切片与跨切片合并
#   --skip-export    : 复用 export_3d_slices.py 已导出的逐切片输入
# 脚本末尾「步骤 4」仍会以 spark,spagcn,spaseg 三方法做 3D 评估 + 空间可视化，
# 其中 spark/spagcn 直接读取已有合并排名 CSV。
#
# 前置：把修复后的 src/py_models/run_spaSEG.py、src/py_models/run_spaGCN.py 同步到 HPC。
# 提交（登录节点）：sbatch src/pipeline/sbatch/Slide_seq_OB2_3D_spaseg_cpu.sh
set -euo pipefail
cd ~/svg_methods
export PATH=$HOME/miniforge3/bin:$PATH
export SVG_PYTHON=$HOME/svg_methods/envs/spatial/bin/python
export SVG_RSCRIPT=$HOME/svg_methods/envs/spatial_R/bin/Rscript
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-36}

bash src/pipeline/run_3d_slices_benchmark.sh \
  --dataset Slide_seq_OB2_3D \
  --methods spaseg \
  --device cpu \
  --skip-export
