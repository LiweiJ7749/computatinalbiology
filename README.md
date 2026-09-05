# SVG 检测方法对比（空间可变基因）

四种空间可变基因（Spatially Variable Gene, SVG）检测方法的批量对比流水线：
**SPARK-X**、**nnSVG**、**SpaGCN**、**SpaSEG**。

面向 Linux/HPC 大规模空间转录组数据处理，把「单一样本、手改路径」的脚本重构为
「一个 run 配置驱动、批量调用」的流水线：统一的共同前处理 → 四方法并行/串行执行 →
统一评估。

## 实验目的

1. 在统一的输入（同一份 h5ad、同一组空间坐标）与统一的输出格式下，批量运行四种
   SVG 方法，保证结果可比。
2. 支持多种空间转录组技术（STARmap / MERFISH / Visium / Slide-seq / stereo-seq 等）
   与 2D/3D 数据。
3. 输出统一的排名 CSV（`gene, stat, pval, padj, rank`），便于方法间一致性与性能对比。
4. 在 Linux/HPC 上可复现构建环境、可规模化调度（CPU 并行 / GPU 深度学习）。

## 四种方法

| 方法 | 语言 | 原理 | 空间维度 |
|---|---|---|---|
| SPARK-X | R | 无参数核检验（投影 / Gaussian / cosine 核） | 2D + 3D |
| nnSVG  | R | 高斯过程（BRISC 逐基因空间方差模型） | 2D |
| SpaGCN | Python | 图卷积网络（GCN），结合表达 + 空间坐标识别空间域 | 2D |
| SpaSEG | Python | 卷积神经网络（CNN）空间域分割 | 2D |

> 3D 数据（如 stereo-seq、Slide-seq 堆叠切片）目前仅 SPARK-X 支持（其 `locus` 为
> `n × d`，天然支持任意维坐标）。

## 运行环境（Linux / HPC 原生）

| 环境（项目内前缀） | 用途 | 包管理 |
|---|---|---|
| `envs/spatial`（Python 3.9） | SpaGCN / SpaSEG + 前处理 | conda + pip（`requirements.txt`），torch 自动探测 CUDA |
| `envs/spatial_R`（R 4.3.1 / Bioc 3.18） | SPARK-X / nnSVG | conda + renv（`renv.lock`） |

- 两个 conda 环境用 `conda create -p` 建在项目 `envs/` 下，由
  [setup_linux_env.sh](setup_linux_env.sh) 一键构建。conda 环境含绝对路径、不可
  跨机器搬移，WSL / HPC 各机器在各自项目目录执行该脚本即可原地重建。
- 流水线脚本 [models_benchmark.sh](src/pipeline/models_benchmark.sh) 会自动探测
  上述环境中的解释器，也支持用 `SVG_PYTHON` / `SVG_RSCRIPT` 显式覆盖。
- R 方法需在项目根目录运行以触发 `.Rprofile` 自动激活 renv。
- 外部方法源码（SpaGCN / SpaSEG 的本地补丁版）随仓库存放在 `src/vendor/`，与
  解释器环境解耦（`src.external_src_dir`，可用 `SVG_EXT_DIR` 覆盖）。

## 快速开始

### 1. 构建环境

前置：已安装 conda/mamba 且 `conda` 在 `PATH` 中（conda ≥ 23 默认启用 libmamba solver）。

```bash
bash setup_linux_env.sh                    # 同时构建 Python + R（默认）
bash setup_linux_env.sh --python-only      # 仅 Python（SpaGCN / SpaSEG）
bash setup_linux_env.sh --r-only           # 仅 R（SPARK-X / nnSVG）
bash setup_linux_env.sh --device cpu       # 强制 CPU 版 torch（默认 auto 自动探测 CUDA）
```

`--device` 取值：`auto`（默认，有 `nvidia-smi` 则装 CUDA 版 torch，否则 CPU 版）、
`cuda`、`cpu`。

### 2. 运行流水线

```bash
# 脚本会自动探测项目 envs/spatial 与 envs/spatial_R（也可用环境变量 SVG_PYTHON/SVG_RSCRIPT）
bash src/pipeline/models_benchmark.sh

# 指定数据集 / 方法子集
bash src/pipeline/models_benchmark.sh --dataset MERFISH_Moffitt
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap --methods spagcn,nnsvg

# 指定核数（nnSVG 并行）与设备（Python 深度学习模型）
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap \
  --methods nnsvg --cores 32 --device cuda

# 只跑前处理（跳过方法），或只跑方法（复用已生成数据）
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap --methods spark --skip-eval
```

结果写入 `results/local_results/<dataset>/<方法子目录>/`，日志在
`results/local_results/<dataset>/logs/`。

### 3. 不激活环境、直接指定解释器

```bash
SVG_PYTHON=$PWD/envs/spatial/bin/python \
SVG_RSCRIPT=$PWD/envs/spatial_R/bin/Rscript \
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap
```

## 本地 vs HPC 运行

### 本地运行（小规模数据集）

本地构建好 `envs/` 后，直接调用主控脚本即可（自动跑完前处理 → 四方法 → 评估）：

```bash
# 单个数据集全流程（默认四方法）
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap

# STARmap AD 8 复本（逐个运行）
bash src/pipeline/models_benchmark.sh --dataset STARmap_AD_13m_ctrl_rep1
bash src/pipeline/models_benchmark.sh --dataset STARmap_AD_13m_disease_rep2

# 指定方法子集 / nnSVG 核数
bash src/pipeline/models_benchmark.sh \
  --dataset Visium_Mouse_Olfactory_Bulb --methods spark,nnsvg --cores 8
```

本地可全流程运行的数据集见「数据集」节的「本地运行」。

### HPC 运行（大规模数据集）

HPC 需先装 Miniforge3、上传代码/数据，并在计算节点重建环境，然后按数据集提交
SBATCH 脚本（位于 `src/pipeline/sbatch/`）：

```bash
# 1) 环境构建（首次，CPU 节点，conda 前缀 envs/ + renv 库）
sbatch src/pipeline/sbatch/build_env.sh

# 2) 冒烟测试（normal_test 队列，验证环境 + 前处理 + SPARK-X）
sbatch src/pipeline/sbatch/test.sh

# 3) 大数据集全流程示例：Visium_HD_Mouse_Kidney
#    前处理 → cpu(SPARK/nnSVG) 与 gpu(SpaSEG) 并行 → 评估
Jp=$(sbatch src/pipeline/sbatch/Visium_HD_Mouse_Kidney_preprocess.sh | awk '{print $4}')
Jc=$(sbatch --dependency=afterok:$Jp src/pipeline/sbatch/Visium_HD_Mouse_Kidney_cpu.sh | awk '{print $4}')
Jg=$(sbatch --dependency=afterok:$Jp src/pipeline/sbatch/Visium_HD_Mouse_Kidney_gpu.sh | awk '{print $4}')
sbatch --dependency=afterok:$Jc,$Jg src/pipeline/sbatch/eval.sh Visium_HD_Mouse_Kidney
```

HPC 各数据集与脚本的对应关系、队列选择见下方「HPC 提交说明」。

## 脚本参数

### models_benchmark.sh

| 参数 | 说明 |
|---|---|
| `--dataset KEY` | 数据集 key（默认 `mouse_brain_STARmap`，见 `configs/datasets.json`） |
| `--h5ad PATH` | 输入 h5ad（覆盖 dataset 注册值） |
| `--spatial PATH` | 坐标文件（`tissue_positions.csv` / `spatial.tar.gz`，可选） |
| `--outdir PATH` | 输出根目录（默认 `results/local_results/<dataset>`） |
| `--sample NAME` | 样本标签（默认取 dataset 注册值） |
| `--methods LIST` | 逗号分隔方法子集（默认全部 `spark,nnsvg,spagcn,spaseg`） |
| `--cores N` | nnSVG 并行线程数（仅 Linux HPC；默认按核数与可用内存自适应） |
| `--device DEV` | 深度学习设备 `auto/cuda/cpu`（默认 `auto`） |
| `--skip-preprocess` | 跳过共同前处理（要求数据已生成） |
| `--skip-eval` | 跳过最后的 `evaluation.py` 汇总 |

### 环境变量

| 变量 | 作用 | 默认 |
|---|---|---|
| `SVG_PYTHON` | 覆盖 Python 解释器 | 自动探测项目 `envs/spatial` |
| `SVG_RSCRIPT` | 覆盖 Rscript 路径 | 自动探测项目 `envs/spatial_R` |
| `CONDA` | `setup_linux_env.sh` 用的 conda 命令 | `conda` |
| `DEVICE` | `setup_linux_env.sh` torch 设备（auto/cuda/cpu） | `auto` |
| `ENVS_DIR` | `setup_linux_env.sh` conda 前缀根目录 | 项目 `envs/` |

## 数据集

`configs/datasets.json` 注册的数据集 key（共 19 个），按运行位置划分：

### 本地运行（小规模）

| key | 技术 / 维度 |
|---|---|
| mouse_brain_STARmap | STARmap / 2D（默认） |
| STARmap_AD_13m_ctrl_rep1 / rep2 | STARmap / 2D |
| STARmap_AD_13m_disease_rep1 / rep2 | STARmap / 2D |
| STARmap_AD_8m_ctrl_rep1 / rep2 | STARmap / 2D |
| STARmap_AD_8m_disease_rep1 / rep2 | STARmap / 2D |
| Visium_Mouse_Olfactory_Bulb | Visium / 2D |
| DLPFC_151507 / 151508 / 151509 | 10x Visium / 2D |

### HPC 运行（大规模）

| key | 技术 / 维度 | 说明 |
|---|---|---|
| DLPFC_151510 | 10x Visium / 2D | 保留用于 HPC 运行测试 |
| Visium_HD_Mouse_Kidney | Visium HD / 2D | 50 万 bin |
| Visium_HD_Human_Breast_Cancer | Visium HD / 2D | 需先在 HPC 转换 feature_slice.h5 |
| MERFISH_Moffitt | MERFISH / 2D | 103 万 spot |
| Stereo_seq_drosophila | stereo-seq / 3D | 仅 SPARK-X |
| Slide_seq_OB2_3D | Slide-seq / 3D | 仅 SPARK-X，3D 重建待实现 |

## 项目结构

```
.
├── setup_linux_env.sh                 # Linux/HPC 一键环境构建（conda + renv）
├── requirements.txt                   # Python 依赖
├── renv.lock                          # R 依赖锁文件
├── .Rprofile                          # 进入项目根自动挂载 renv 项目库
├── configs/
│   ├── datasets.json                  # 数据集注册表
│   ├── run_params.json                # 每数据集的差异化运行参数（nnSVG 过滤/SpaGCN 域数等）
│   └── model_params/                  # 各方法超参数
├── data/                              # 输入 h5ad（各技术子目录）
├── envs/                              # conda 前缀（gitignore，由 setup 脚本重建）
├── results/local_results/             # 输出（每数据集一目录）
└── src/
    ├── __init__.py                    # 路径/常量/数据集注册/共同前处理核心
    ├── pipeline/
    │   ├── models_benchmark.sh        # 主控流水线脚本（Bash，本地/HPC 通用）
    │   └── sbatch/                    # HPC 各数据集的 SBATCH 提交脚本
    ├── preprocess/
    │   ├── h5ad_preprocess.py         # 共同前处理 CLI
    │   └── 10xVisium_pretreat_h5toh5ad.py  # Visium HD feature_slice.h5 -> h5ad 转换
    ├── r_models/
    │   ├── run_spark.r                # SPARK-X
    │   └── run_nnSVG.r                # nnSVG
    ├── py_models/
    │   ├── run_spaGCN.py              # SpaGCN
    │   └── run_spaSEG.py              # SpaSEG
    ├── vendor/                        # SpaGCN_src / SpaSEG_src 运行源码
    └── utils/evaluation.py            # 评估指标汇总
```

## HPC 提交说明

### SBATCH 脚本清单（`src/pipeline/sbatch/`）

| 脚本 | 数据集 | 队列 |
|---|---|---|
| build_env.sh | 通用（建环境） | 7542-64C-512G |
| test.sh | 冒烟测试（normal_test） | normal_test |
| eval.sh | 通用评估，`sbatch eval.sh <dataset>` | 7542-64C-512G |
| DLPFC_151510_cpu/gpu.sh | DLPFC_151510（测试保留） | 7542 / gpu_v100 |
| Visium_HD_Mouse_Kidney_{preprocess,cpu,gpu}.sh | Visium_HD_Mouse_Kidney | 7542 / gpuB |
| Visium_HD_Human_Breast_Cancer_{convert,preprocess,cpu}.sh | Human_Breast_Cancer | 6126-24C-768G |
| MERFISH_Moffitt_{preprocess,cpu,gpu}.sh | MERFISH | 6126-24C-768G / gpuB |
| Stereo_seq_drosophila_cpu.sh | Stereo_seq（3D） | 6126-24C-768G |
| Slide_seq_OB2_3D_cpu.sh | Slide_seq（3D） | 6126-24C-768G |

### 队列选型原则

- **CPU**：spot <1 万 → `6240-36C-192G`；1 万~50 万 → `7542-64C-512G`；
  dense 大矩阵 / 50 万+ → `6126-24C-768G`。
- **GPU**：spot <1 万 → `gpu_v100`（32G）；≥1 万 → `gpuB`（80G）。
- **SpaGCN** 邻接矩阵 O(n²)，>几万 spot 不适用（Visium_HD / MERFISH 跳过 SpaGCN）。

### 其他说明

1. conda 环境含绝对路径、不可跨节点搬移，需在 HPC 各节点分别执行
   `bash setup_linux_env.sh` 原地重建（代码随项目目录迁移）。
2. 平台 Slurm 队列、存储配额、作业模板等合规要求见 [docs/HPC_note.md](docs/HPC_note.md)。
3. nnSVG 默认并行核数按「核数 ∩ 可用内存（每 worker 约 1.5GB）」自适应；大内存节点
   显式传 `--cores N`。
4. 数据集级运行参数（nnSVG 过滤阈值等）在 `configs/run_params.json`，由
   `models_benchmark.sh` 读取并注入环境变量。