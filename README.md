# SVG 检测方法对比

四种空间可变基因（SVG）检测方法的批量对比流水线：
**SPARK-X**、**nnSVG**、**SpaGCN**、**SpaSEG**。

## 运行环境

| 环境 | 用途 | 路径 |
|---|---|---|
| **renv** (R 4.4.3) | R 包管理 (SPARK, nnSVG) | `D:\R-4.4.3\bin\Rscript.exe`，`.Rprofile` 自动激活 |
| **env_spatial** | Python 方法 (SpaGCN, SpaSEG) | `env_spatial\python.exe` |
| **env_R** | conda R 备选 | `env_R\lib\R\bin\Rscript.exe`（回退） |

## 使用方法

### 完整流水线（四方法 + 前处理 + 评估）

**PowerShell（推荐，Windows 原生）：**

```powershell
# 默认数据集 mouse_brain_STARmap
.\src\pipeline\run_benchmark.ps1

# 指定数据集
.\src\pipeline\run_benchmark.ps1 -dataset mouse_brain_STARmap

# 指定方法子集
.\src\pipeline\run_benchmark.ps1 -dataset mouse_brain_STARmap -methods spagcn,spaseg

# 自定义输出目录
.\src\pipeline\run_benchmark.ps1 -outdir .\results\my_run -sample my_sample
```

**Bash（需 WSL / Git Bash）：**

```bash
# 默认数据集 mouse_brain_STARmap
bash src/pipeline/models_benchmark.sh

# 指定数据集
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap

# 指定方法子集
bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap --methods spagcn,spaseg

# 自定义输入
bash src/pipeline/models_benchmark.sh --h5ad ./data/xxx.h5ad --outdir ./results/my --sample my
```

### 分步运行

```bash
# 1) 共同前处理
python src/preprocess/h5ad_preprocess.py --dataset mouse_brain_STARmap

# 2) R 方法（在项目根目录运行，renv 自动激活）
Rscript src/r_models/run_spark.r  <outdir>/SPARK_X  <sample>
Rscript src/r_models/run_nnSVG.r  <outdir>/nnSVG    <sample>

# 3) Python 方法
python src/py_models/run_spaGCN.py --dataset mouse_brain_STARmap
python src/py_models/run_spaSEG.py --dataset mouse_brain_STARmap

# 4) 评估
python src/utils/evaluation.py --dataset mouse_brain_STARmap
```

### 环境变量

- `SVG_PYTHON` — 指定 Python 解释器（默认 `env_spatial/python.exe`）
- `SVG_RSCRIPT` — 指定 Rscript 路径（默认 `D:\R-4.4.3\bin\Rscript.exe`）

## 项目结构

```
src/
├── pipeline/models_benchmark.sh   # 主控流水线脚本
├── preprocess/h5ad_preprocess.py  # 共同前处理
├── r_models/
│   ├── run_spark.r                # SPARK-X
│   └── run_nnSVG.r                # nnSVG
├── py_models/
│   ├── run_spaGCN.py              # SpaGCN
│   └── run_spaSEG.py              # SpaSEG
└── utils/evaluation.py            # 评估指标汇总
```
