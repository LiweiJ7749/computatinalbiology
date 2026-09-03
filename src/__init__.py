# -*- coding: utf-8 -*-
"""src —— 批量 SVG 方法对比：运行配置解析 / 目录管理 / 共同前处理

把四种方法（SPARK-X、nnSVG、SpaGCN、SpaSEG）从"单一样本、手改路径"的脚本
重构成"一个 run 配置驱动、被 pipeline 批量调用"的形式，所有共同逻辑集中于此：

  1) 初始化 / 常量
       ROOT / DATA_DIR / RESULTS_DIR / METHOD_SUBDIRS / DATASETS ...
  2) 运行配置（run 字典）
       resolve_run()   把 dataset / h5ad / outdir / sample 归一成唯一配置
       ensure_run_dirs()  目录检查与创建（含每个方法的子目录）
  3) 四方法共同前处理（原 src/preprocess/h5ad_preprocess.py 的逻辑上移到这里，
     该脚本只保留一个调用本模块的薄 CLI）
       locate_spatial / load_coords / export_r_format / prepare_spagcn /
       prepare_spaseg / preprocess_run()
  4) 环境探测
       find_python() / find_rscript()

约定（方法名 -> 结果子目录 / 运行脚本 / 输出 CSV 前缀）::

    spark  -> SPARK_X/ , src/r_models/run_spark.r   , SVG_SPARK_<sample>.csv
    nnsvg  -> nnSVG/   , src/r_models/run_nnSVG.r   , SVG_nnSVG_<sample>.csv
    spagcn -> spaGCN/  , src/py_models/run_spaGCN.py, SVG_spaGCN_<sample>.csv
    spaseg -> spaSEG/  , src/py_models/run_spaSEG.py, SVG_spaSEG_<sample>.csv

其中 <sample> 为 run["sample"]（默认取自数据集注册表 DATASETS 的 sample 字段）。

设计要点：模块顶层只 import 标准库与纯 pathlib，anndata/numpy/pandas/scipy 等
重依赖均在函数内延迟导入 —— 模型脚本执行 ``import src`` 时开销极低。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1) 常量路径
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # 项目根
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "local_results"
SRC_DIR = ROOT / "src"

# 本机 conda 前缀里的解释器（Windows）；HPC/Linux 上不存在时由 find_* 兜底探测
ENV_PYTHON = ROOT / "env_spatial" / "python.exe"
ENV_RSCRIPT = ROOT / "env_R" / "lib" / "R" / "bin" / "Rscript.exe"

# ---------------------------------------------------------------------------
# 2) 方法注册（子目录 / 脚本 / 标准顺序）
# ---------------------------------------------------------------------------
METHOD_SUBDIRS = {"spark": "SPARK_X", "nnsvg": "nnSVG",
                  "spagcn": "spaGCN", "spaseg": "spaSEG"}
ALL_METHODS = ["spark", "nnsvg", "spagcn", "spaseg"]   # 标准执行顺序

MODEL_SCRIPTS = {
    "spark":  SRC_DIR / "r_models" / "run_spark.r",
    "nnsvg":  SRC_DIR / "r_models" / "run_nnSVG.r",
    "spagcn": SRC_DIR / "py_models" / "run_spaGCN.py",
    "spaseg": SRC_DIR / "py_models" / "run_spaSEG.py",
}

# 统一方法颜色（ColorBrewer Set1 前四色，Python 绘图 / evaluation 使用；
# R 脚本中无法 import 本模块，在脚本内硬编码同一组 hex 以保证跨语言一致）。
METHOD_COLORS = {
    "spark":  "#E41A1C",   # 红
    "nnsvg":  "#377EB8",   # 蓝
    "spagcn": "#4DAF4A",   # 绿
    "spaseg": "#984EA3",   # 紫
}
METHOD_LABELS = {
    "spark":  "SPARK-X",
    "nnsvg":  "nnSVG",
    "spagcn": "SpaGCN",
    "spaseg": "SpaSEG",
}

# ---------------------------------------------------------------------------
# 3) 数据集注册表（可被命令行显式参数覆盖）
# ---------------------------------------------------------------------------
DATASETS = {
    "mouse_brain_STARmap": {
        "h5ad": (DATA_DIR / "STARmap" / "mouse_brain_cortex"
                 / "mouse_brain_STARmap_processed.h5ad"),
        "sample": "STARmap_Mouse_Brain",
    },
    "Visium_Mouse_Olfactory_Bulb": {
        "h5ad": DATA_DIR / "Visium" / "Mouse_Olf_Bulb"
                / "Visium_Mouse_Olfactory_Bulb.h5ad",
        "spatial": (DATA_DIR / "Visium" / "Mouse_Olf_Bulb"
                    / "Visium_Mouse_Olfactory_Bulb_spatial.tar.gz"),
        "sample": "Visium_Mouse_Olfactory_Bulb",
    },
}


# ---------------------------------------------------------------------------
# 4) run 配置解析 + 目录管理
# ---------------------------------------------------------------------------
def resolve_run(dataset=None, h5ad=None, spatial=None, outdir=None,
                sample=None, methods=None) -> dict:
    """把一个 run 的意图归一成唯一的配置字典。

    参数可混用：命中 ``dataset`` 注册表后，显式传入的 h5ad/spatial/outdir/sample
    会覆盖注册表默认值。返回字典字段：
      dataset / h5ad / spatial / outdir / sample / methods / method_dirs
    """
    methods = list(methods or ALL_METHODS)
    bad = [m for m in methods if m not in METHOD_SUBDIRS]
    if bad:
        raise ValueError(f"未知方法 {bad}，可选: {list(METHOD_SUBDIRS)}")
    methods = [m for m in ALL_METHODS if m in methods]   # 规范顺序

    reg = DATASETS.get(dataset, {}) if dataset else {}

    # --- h5ad ---
    h5ad_path = Path(h5ad) if h5ad else reg.get("h5ad")
    if h5ad_path is not None and not h5ad_path.is_absolute():
        cand = ROOT / h5ad_path
        h5ad_path = cand if cand.exists() else h5ad_path
    if h5ad_path is None:
        raise ValueError("无法确定 h5ad：请提供 --dataset 或 --h5ad，"
                         f"已知数据集: {list(DATASETS)}")
    h5ad_path = Path(h5ad_path)

    # --- spatial / outdir / sample ---
    spatial_path = Path(spatial) if spatial else reg.get("spatial")
    out_root = Path(outdir) if outdir else reg.get(
        "outdir", RESULTS_DIR / (dataset or h5ad_path.stem))
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    sample_name = sample or reg.get("sample") or h5ad_path.stem

    return {
        "dataset": dataset,
        "h5ad": h5ad_path,
        "spatial": spatial_path,
        "outdir": out_root,
        "sample": sample_name,
        "methods": methods,
        "method_dirs": {m: out_root / METHOD_SUBDIRS[m] for m in methods},
    }


def ensure_run_dirs(run: dict, methods=None) -> None:
    """检查并创建 run 输出目录与每个方法的子目录。"""
    methods = list(methods) if methods is not None else list(run["methods"])
    run["outdir"].mkdir(parents=True, exist_ok=True)
    for m in methods:
        run["method_dirs"][m].mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 5) 环境探测（供 pipeline .sh 或交互式调用）
# ---------------------------------------------------------------------------
def find_python():
    """返回可用的 python 解释器路径（优先本项目 env_spatial）。"""
    if ENV_PYTHON.exists():
        return str(ENV_PYTHON)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_rscript():
    """返回可用的 Rscript 路径（优先本项目 env_R）。"""
    if ENV_RSCRIPT.exists():
        return str(ENV_RSCRIPT)
    p = shutil.which("Rscript")
    return p


# ---------------------------------------------------------------------------
# 6) 共同前处理（延迟导入 anndata/scipy 等重依赖）
# ---------------------------------------------------------------------------
def _read_tissue_positions(source: Path):
    """读取 tissue_positions.csv（或其所在 spatial.tar.gz），返回以 barcode 为索引的 DataFrame。"""
    import io

    import pandas as pd

    if source.suffix in (".gz", ".tgz") or ".tar" in source.name:
        import tarfile

        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_positions.csv")), None)
            if member is None:
                raise FileNotFoundError(f"{source} 内未找到 tissue_positions.csv")
            raw = tar.extractfile(member).read()
        tp = pd.read_csv(io.BytesIO(raw))
    else:
        tp = pd.read_csv(source)
    if "barcode" not in tp.columns:
        raise ValueError("tissue_positions.csv 缺少 barcode 列")
    return tp.set_index("barcode")


def _read_hires_image(source: Path):
    """从 spatial.tar.gz 提取 tissue_hires_image.png（BGR 数组），无则 None。"""
    try:
        import tarfile

        import numpy as np

        with tarfile.open(source, "r:*") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith("tissue_hires_image.png")), None)
            if member is None:
                return None
            data = tar.extractfile(member).read()
        import cv2

        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def locate_spatial(h5ad_path: Path, spatial_arg) -> tuple:
    """按优先级返回 (tissue_positions.csv 路径 | None, spatial.tar.gz 路径 | None)。"""
    if spatial_arg is not None:
        s = Path(spatial_arg)
        if s.exists():
            return (s, None) if s.suffix.lower() == ".csv" else (None, s)
    # 已解压的全局坐标表
    extracted = RESULTS_DIR / "spatial" / "tissue_positions.csv"
    if extracted.exists():
        return extracted, None
    # h5ad 同目录下的 *_spatial.tar.gz
    tgz = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
    if tgz.exists():
        return None, tgz
    return None, None


def load_coords(h5ad, spatial_arg) -> "pd.DataFrame":
    """读入坐标并与 h5ad 的 obs.index 对齐，返回 barcode 为索引的 DataFrame。

    Visium：x/y（像素坐标），若存在保留 array_row/array_col；
    STARmap 等无坐标表的平台：回退 h5ad.obsm["spatial"]（前两列 -> x/y）。
    """
    import numpy as np
    import pandas as pd

    h5ad_path = Path(h5ad.filename) if h5ad.filename else None
    tp = None
    if h5ad_path is not None:
        csv_path, tgz_path = locate_spatial(h5ad_path, spatial_arg)
        if csv_path is not None:
            tp = _read_tissue_positions(csv_path)
        elif tgz_path is not None:
            tp = _read_tissue_positions(tgz_path)

    if tp is not None and len(tp) > 0:
        if "in_tissue" in tp.columns:
            tp = tp[tp["in_tissue"] == 1]
        tp = tp.loc[[b for b in h5ad.obs.index if b in tp.index]]
        if "pxl_col_in_fullres" in tp.columns and "pxl_row_in_fullres" in tp.columns:
            tp = tp.rename(columns={"pxl_col_in_fullres": "x",
                                    "pxl_row_in_fullres": "y"})
        if len(tp) > 0:
            return tp

    if "spatial" in h5ad.obsm and h5ad.obsm["spatial"] is not None:
        coords = np.asarray(h5ad.obsm["spatial"], dtype=float)
        return pd.DataFrame(coords[:, :2], index=h5ad.obs.index,
                            columns=["x", "y"])
    raise FileNotFoundError("找不到坐标来源"
                            "(tissue_positions.csv / spatial.tar.gz / obsm['spatial'])")


def _raw_counts_matrix(h5ad):
    """返回喂给 R 方法的"真实 counts"矩阵 (spots x genes)，优先 layers['raw_count']。"""
    import numpy as np
    from scipy import sparse

    if "raw_count" in h5ad.layers and h5ad.layers["raw_count"] is not None:
        X = h5ad.layers["raw_count"]
        print("      [表达矩阵] 使用 layers['raw_count']（真实 counts）")
    else:
        X = h5ad.X
        print("      [表达矩阵] 无 raw_count 层，回退 X（注意：可能不是原始 counts）")
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    return X.astype(np.float64)


def export_r_format(h5ad, tp, outdir: Path) -> None:
    """导出 SPARK-X / nnSVG 可直接读取的 counts.mtx / genes.csv / barcodes.csv / location.csv。

    counts.mtx 为 genes x spots（行为基因）。优先 raw_count 层（真实 counts），
    与 h5ad_preprocess 原 docstring 一致；无 raw_count 时才回退 h5ad.X。
    """
    import pandas as pd
    from scipy import io as sio

    outdir.mkdir(parents=True, exist_ok=True)

    X = _raw_counts_matrix(h5ad)
    gene_names = list(h5ad.var.index.astype(str))

    barcodes = [b for b in h5ad.obs.index if b in tp.index]
    keep_idx = [i for i, b in enumerate(h5ad.obs.index) if b in tp.index]
    X = X[keep_idx, :]                     # spots x genes（与坐标对齐）
    tp_aligned = tp.loc[barcodes]
    X = X.T.tocsr()                        # genes x spots（R 方法约定）

    sio.mmwrite(str(outdir / "counts.mtx"), X, field="integer")
    pd.Series(gene_names).to_csv(outdir / "genes.csv", index=False, header=False)
    pd.Series(barcodes).to_csv(outdir / "barcodes.csv", index=False, header=False)

    loc = pd.DataFrame({
        "x": tp_aligned["x"].values, "y": tp_aligned["y"].values,
    }, index=barcodes)
    loc.to_csv(outdir / "location.csv", index_label="barcode")
    print(f"      [R 格式] {X.shape[0]} genes x {X.shape[1]} spots -> {outdir}")


def prepare_spagcn(h5ad, tp, img, outdir: Path, sample_name: str) -> None:
    """为 SpaGCN 准备输入：写 <sample>_spaGCN.h5ad + 坐标列 + 组织学图像。

    SpaGCN 官方要求的 obs 坐标列（对应 positions.txt 的 6 列）：
      x_array = array_row,  y_array = array_col
      x_pixel = pxl_row_in_fullres, y_pixel = pxl_col_in_fullres
    组织学图像可选（histology=False 时 SpaGCN 仅用空间坐标建邻接矩阵）。
    """
    import anndata as ad

    outdir.mkdir(parents=True, exist_ok=True)
    adata = _copy_h5ad(h5ad)
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    adata.obs["x_pixel"] = tp_aligned["x"].astype(float).values
    adata.obs["y_pixel"] = tp_aligned["y"].astype(float).values
    if "array_row" in tp_aligned.columns and "array_col" in tp_aligned.columns:
        adata.obs["x_array"] = tp_aligned["array_row"].astype(int).values
        adata.obs["y_array"] = tp_aligned["array_col"].astype(int).values
    else:
        adata.obs["x_array"] = tp_aligned["x"].astype(int).values
        adata.obs["y_array"] = tp_aligned["y"].astype(int).values

    h5ad_out = outdir / f"{sample_name}_spaGCN.h5ad"
    adata.write(h5ad_out)
    if img is not None:
        import cv2

        cv2.imwrite(str(outdir / "histology.png"), img)
        print(f"      [SpaGCN] 坐标列 + 组织学图像 -> {h5ad_out} / histology.png")
    else:
        print(f"      [SpaGCN] 坐标列（无组织学图像，可用 histology=False）-> {h5ad_out}")
    del adata


def prepare_spaseg(h5ad, tp, outdir: Path, sample_name: str) -> None:
    """为 SpaSEG 准备输入：写 <sample>_spaSEG.h5ad + obsm['spatial'] 坐标。"""
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    adata = _copy_h5ad(h5ad)
    barcodes = list(adata.obs.index)
    tp_aligned = tp.loc[[b for b in barcodes if b in tp.index]]

    adata.obsm["spatial"] = np.column_stack([
        tp_aligned["x"].astype(float).values,
        tp_aligned["y"].astype(float).values,
    ]).astype(np.float32)

    h5ad_out = outdir / f"{sample_name}_spaSEG.h5ad"
    adata.write(h5ad_out)
    print(f"      [SpaSEG] obsm['spatial'] -> {h5ad_out}")
    del adata


def _copy_h5ad(h5ad):
    """复制一个轻量 AnnData（避免污染读入的原始对象），保留 layers（如 raw_count）。"""
    import anndata as ad

    return ad.AnnData(
        X=h5ad.X.copy(),
        obs=h5ad.obs.copy(),
        var=h5ad.var.copy(),
        layers={k: v.copy() for k, v in h5ad.layers.items()},
        obsm={k: v.copy() for k, v in h5ad.obsm.items()},
        uns={k: v for k, v in h5ad.uns.items()},
    )


def histology_source(run: dict):
    """返回可提取组织学图像的 spatial.tar.gz 路径（无则 None）。"""
    sp = run.get("spatial")
    if sp is not None:
        s = Path(sp)
        if s.exists() and s.suffix.lower() != ".csv":
            return s
    h5ad_path = run["h5ad"]
    tgz = h5ad_path.with_name(h5ad_path.name.replace(".h5ad", "_spatial.tar.gz"))
    return tgz if tgz.exists() else None


def preprocess_run(run: dict, methods=None) -> dict:
    """对一次 run 生成四个方法所需的全部输入，并建立目录。

    - R 方法（spark/nnsvg）：写各方法子目录下的 counts.mtx / genes.csv / barcodes.csv / location.csv
    - SpaGCN / SpaSEG：写 <sample>_spaGCN.h5ad / <sample>_spaSEG.h5ad（含坐标）
    """
    import anndata as ad

    methods = [m for m in (methods or run["methods"])]
    ensure_run_dirs(run, methods=methods)

    h5ad_path = run["h5ad"]
    if not h5ad_path.exists():
        raise FileNotFoundError(f"h5ad 不存在: {h5ad_path}")
    print(f"[preprocess] 读取 h5ad: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path)
    print(f"      shape = {adata.shape} (spots x genes)")

    tp = load_coords(adata, run.get("spatial"))
    print(f"      对齐坐标后 spots = {len(tp)}")

    img = None
    if "spagcn" in methods:
        src = histology_source(run)
        if src is not None:
            img = _read_hires_image(src)
            if img is not None:
                print("      已提取组织学图像 (tissue_hires_image.png)")

    for m in methods:
        outdir = run["method_dirs"][m]
        if m in ("spark", "nnsvg"):
            export_r_format(adata, tp, outdir)
        elif m == "spagcn":
            prepare_spagcn(adata, tp, img, outdir, run["sample"])
        elif m == "spaseg":
            prepare_spaseg(adata, tp, outdir, run["sample"])
    del adata
    print("[preprocess] 完成 ✓")
    return run


# 兼容：一些旧脚本/文档把它当作 dict 使用
SUBDIRS = METHOD_SUBDIRS


def __getattr__(name):
    """对旧引用给出更友好的错误（避免 import 时即失败）。"""
    raise AttributeError(
        f"src 模块没有属性 {name!r}。批量重构后请使用 "
        "src.resolve_run / src.ensure_run_dirs / src.preprocess_run。")


if __name__ == "__main__":      # python -m src 无意义时的提示（防止静默）
    print(__doc__)
