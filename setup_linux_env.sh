#!/usr/bin/env bash
# =============================================================================
# setup_linux_env.sh —— 在 Linux / WSL / HPC 上构建本项目运行时环境（仅 Linux）
# =============================================================================
# 项目为纯 Linux/HPC 工作流（Windows 支持已移除）。本脚本在项目目录内创建
# conda 环境（conda create -p），并把四方法统一跑在 `envs/` 前缀下，便于随
# 项目目录一起管理 / 在 HPC 各节点原地重建。
#
# 产物（都在项目目录下，路径随项目迁移）：
#   - envs/spatial    ：Python 3.9 + 全部 Python 依赖（SpaGCN / SpaSEG + 前处理）
#   - envs/spatial_R  ：R 4.3.1 + r-recommended + renv 管理的 R 包（SPARK-X/nnSVG，Bioc 3.18）
#   - src/vendor/     ：SpaGCN_src / SpaSEG_src 运行所需纯 Python 源码（已随仓库纳入版本控制）
#
# 依赖：已安装的 conda/mamba（conda>=23 默认启用 libmamba solver，速度足够）。
# 说明：conda 环境含绝对路径、不可跨机器搬移；本脚本设计为“可复现”，即在 WSL 或
#       HPC 各节点上分别执行本脚本即可在各自项目目录下原地重建环境。
#
# 用法：
#   bash setup_linux_env.sh              # 同时构建 Python 与 R（默认）
#   bash setup_linux_env.sh --python-only
#   bash setup_linux_env.sh --r-only
#   bash setup_linux_env.sh --device cpu # 强制 CPU 版 torch（默认 auto 自动探测 CUDA）
#
# 环境变量（可选覆盖）：
#   CONDA / DEVICE / ENVS_DIR
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$ROOT"

CONDA="${CONDA:-conda}"
DEVICE="${DEVICE:-auto}"
ONLY="all"
ENVS_DIR="${ENVS_DIR:-$ROOT/envs}"          # conda 前缀根目录（默认项目内 envs/）
PY_PREFIX="$ENVS_DIR/spatial"
R_PREFIX="$ENVS_DIR/spatial_R"
PYBIN="$PY_PREFIX/bin/python"
RSCRIPT="$R_PREFIX/bin/Rscript"

# ---------------- 参数解析 ----------------
while [ $# -gt 0 ]; do
  case "$1" in
    --python-only) ONLY="python"; shift ;;
    --r-only)      ONLY="r"; shift ;;
    --device)      DEVICE="${2:-auto}"; shift 2 ;;
    -h|--help)     sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "[错误] 未知参数: $1" >&2; exit 1 ;;
  esac
done

# ---------------- 工具函数 ----------------
log()     { printf '  [%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
section() { printf '\n========================================\n  [%s] %s\n========================================\n' "$(date '+%H:%M:%S')" "$1"; }

command -v "$CONDA" >/dev/null 2>&1 || { echo "[错误] 找不到 $CONDA，请先安装 miniconda/miniforge" >&2; exit 2; }
CONDA_BASE="$("$CONDA" info --base)"

# ---------------------------------------------------------------------------
# 阶段 1：Python 环境（conda prefix envs/spatial + pip 依赖）
# ---------------------------------------------------------------------------
build_python() {
  section "构建 Python 环境: $PY_PREFIX"
  if [ -x "$PYBIN" ]; then
    log "已存在 $PYBIN，跳过创建（如需重建先 conda env remove -p $PY_PREFIX）"
  else
    "$CONDA" create -p "$PY_PREFIX" python=3.9 pip -y
  fi

  # 探测 CUDA：auto -> 有 nvidia-smi 且可用则装 cu124，否则 CPU
  local use_cuda=0
  if [ "$DEVICE" = "cuda" ]; then
    use_cuda=1
  elif [ "$DEVICE" = "cpu" ]; then
    use_cuda=0
  elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    use_cuda=1
  fi

  log "升级 pip 与基础工具"
  "$PYBIN" -m pip install --upgrade pip wheel
  # setuptools>=81 已移除 pkg_resources，但 numba 0.55.2 / llvmlite 仍依赖它，
  # 故固定 <81（numba import 时 `from pkg_resources import ...`）。
  "$PYBIN" -m pip install "setuptools<81"

  # torch 的 cu124/cpu 轮子体积大；直接用 curl 强制 IPv4 下载本地轮子再安装，
  # 避免部分网络（如 WSL/某些节点）走 IPv6 导致下载极慢。
  local torch_whl=""
  if [ "$use_cuda" -eq 1 ]; then
    torch_whl=/tmp/torch-2.5.1+cu124-cp39-cp39-linux_x86_64.whl
    log "检测到 CUDA，下载 torch==2.5.1+cu124"
    [ -f "$torch_whl" ] || curl -4 -L --retry 3 -o "$torch_whl" \
      "https://mirrors.aliyun.com/pytorch-wheels/cu124/torch-2.5.1%2Bcu124-cp39-cp39-linux_x86_64.whl"
    "$PYBIN" -m pip install "$torch_whl" \
      -f https://mirrors.aliyun.com/pytorch-wheels/cu124/
  else
    torch_whl=/tmp/torch-2.5.1+cpu-cp39-cp39-linux_x86_64.whl
    log "未检测到 CUDA，下载 CPU 版 torch==2.5.1+cpu"
    [ -f "$torch_whl" ] || curl -4 -L --retry 3 -o "$torch_whl" \
      "https://download.pytorch.org/whl/cpu/torch-2.5.1%2Bcpu-cp39-cp39-linux_x86_64.whl"
    "$PYBIN" -m pip install "$torch_whl" \
      -f https://download.pytorch.org/whl/cpu
  fi

  log "安装其余依赖（requirements.txt，跳过 torch 行）"
  grep -vE '^[[:space:]]*(torch|#)|^[[:space:]]*$' requirements.txt > /tmp/svg_reqs.txt
  # 用清华 PyPI 镜像：部分网络下 aliyun PyPI 镜像对 wheel 限速严重
  "$PYBIN" -m pip install -r /tmp/svg_reqs.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple/ || log "[警告] 部分依赖安装失败，将继续"

  # 外部方法源码已 vendor 于 src/vendor/（见 external_src_dir），无需再复制环境目录。
  log "Python 环境就绪: $PYBIN"
  "$PYBIN" -c "import sys, numpy, scanpy, torch; print('  python', sys.version.split()[0], '| numpy', numpy.__version__, '| scanpy', scanpy.__version__, '| torch', torch.__version__)"
}

# ---------------------------------------------------------------------------
# 阶段 2：R 环境（conda prefix envs/spatial_R，renv 管理包）
# ---------------------------------------------------------------------------
build_r() {
  section "构建 R 环境: $R_PREFIX"
  if [ -x "$RSCRIPT" ]; then
    log "已存在 $RSCRIPT，跳过创建"
    # 已存在的环境也补齐系统编译依赖与 R 推荐包（幂等）
    "$CONDA" install -p "$R_PREFIX" -c conda-forge zlib xz imagemagick "r-recommended=4.3" -y || true
  else
    # 匹配 renv.lock 的 R 4.3.1（HPC 平台仅提供 R<=4.3.1，故统一降级）。
    # r-recommended 提供 R 推荐包（Matrix/lattice/MASS/...）：Bioc 源码包编译需
    # 链接 site 库中的 Matrix，故必须与 r-base 一起装，否则 restore 会失败。
    # 一并安装 zlib/xz/imagemagick 开发包（renv::restore 从源码编译 XVector 等
    # Bioc 包需要 zlib.h/lzma.h，magick 包需要 ImageMagick 的 Magick++ 头）。
    log "创建 conda R 环境（r-base 4.3.1 + r-recommended + zlib/xz/imagemagick）"
    "$CONDA" create -p "$R_PREFIX" -c conda-forge "r-base=4.3.1" "r-recommended=4.3" zlib xz imagemagick -y
  fi

  # 激活环境：conda 包安装的 R 默认用前缀编译器
  # (x86_64-conda-linux-gnu-cc/c++/gfortran)，这些只有激活后才会进入 PATH，
  # 否则 renv::restore 从源码编译 C/C++ 包会报 "compiler not found"。
  source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
  conda activate "$R_PREFIX" 2>/dev/null || true

  # SPARK(pkg Makevars 强制 CXX_STD=CXX11) 新版 RcppArmadillo 要求 C++14，
  # 否则报 "C++14 compiler required"。全局把 CXX11STD 提到 C++14（向后兼容 C++11）。
  mkdir -p "$HOME/.R"
  printf 'CXX11STD = -std=gnu++14\n' > "$HOME/.R/Makevars"

  log "安装 renv（写入本 conda 前缀 site 库；项目 .Rprofile 不再走 activate）"
  RENV_CONFIG_AUTOLOADER_ENABLED=FALSE Rscript --vanilla -e 'if (!requireNamespace("renv", quietly=TRUE)) install.packages("renv", repos="https://mirrors.tuna.tsinghua.edu.cn/CRAN")' || true
  log "renv::restore（读取 renv.lock 构建项目库 renv/library/）"
  # 显式 library(renv) + 指定项目根，避免 .Rprofile/activate 联网加载；Bioc 版本
  # 与锁文件保持一致（renv/ 目录被 .gitignore，克隆到 HPC 时 settings.json 不会
  # 随 git 同步，故在此幂等校正为 3.18）。
  ( cd "$ROOT" && Rscript --vanilla -e 'library(renv); renv::settings$bioconductor.version("3.18"); renv::restore(prompt = FALSE)' )

  log "R 环境就绪: $RSCRIPT"
  "$RSCRIPT" -e 'cat("  R", as.character(getRversion()), "| renv", as.character(packageVersion("renv")), "\n")'
}

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
case "$ONLY" in
  python) build_python ;;
  r)      build_r ;;
  all)    build_python; build_r ;;
esac

section "完成"
cat <<EOF
  环境已构建完成（项目内前缀：$ENVS_DIR）。

  运行流水线（建议直接指定解释器，无需激活任何环境）：

    SVG_PYTHON=$PYBIN SVG_RSCRIPT=$RSCRIPT \\
      bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap

  或先激活再运行（脚本也能自动探测到）：

    conda activate $PY_PREFIX
    bash src/pipeline/models_benchmark.sh --dataset mouse_brain_STARmap

  R 方法（SPARK-X / nnSVG）需在项目根目录运行以挂载 renv 项目库（.Rprofile）。
EOF
