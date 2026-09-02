"""10x H5 -> h5ad 预处理脚本。

将 10x 格式的 filtered_feature_bc_matrix.h5 统一转换为通用 .h5ad 文件，
供后续 SVG 检测方法对比使用。

依赖（已在 env_spatial 中安装）：scanpy / anndata / h5py / scipy。

用法（在项目根目录 F:\\computatinalbiology 下）:
    # 转换默认的 Mouse_Olf_Bulb 数据（输出到 data 目录，已被 .gitignore 忽略）
    python ./src/preprocess/Visium_pretreat_h5toh5ad.py

    # 转换任意单个文件。<input.h5> 可为绝对路径、相对当前目录的路径，
    # 或相对 data/ 目录的路径（如 Visium/Mouse_Olf_Bulb/xxx.h5，会自动定位）。
    # 省略 [output.h5ad] 时自动在与输入同目录生成同名 .h5ad。
    python ./src/preprocess/Visium_pretreat_h5toh5ad.py <input.h5> [output.h5ad]

    示例:
    python ./src/preprocess/Visium_pretreat_h5toh5ad.py \\
        ./data/Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb_filtered_feature_bc_matrix.h5 \\
        ./data/Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb.h5ad
"""
import sys
from pathlib import Path

import scanpy as sc

# 项目根目录（本文件位于 src/preprocess/ 下，parents[2] = 项目根）
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

# (输入相对路径, 输出相对路径)，可在此追加更多数据集
CONVERSIONS = [
    (
        "Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb_filtered_feature_bc_matrix.h5",
        "Visium/Mouse_Olf_Bulb/Visium_Mouse_Olfactory_Bulb.h5ad",
    ),
]


def to_h5ad(h5_path: Path, out_path: Path) -> None:
    # read_10x_h5 直接解析 10x 的 matrix/filtered_feature_bc_matrix.h5
    adata = sc.read_10x_h5(h5_path)
    # 基因名可能存在重复，去重以保证 var 索引唯一
    adata.var_names_make_unique()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write(out_path)
    print(
        f"OK: {h5_path.name} -> {out_path} "
        f"({adata.n_obs} cells x {adata.n_vars} genes)"
    )


def main() -> None:
    if len(sys.argv) >= 2:
        # 输入：优先用给定路径；不存在则尝试在 data/ 目录下定位
        inp = Path(sys.argv[1])
        if not inp.exists():
            cand = DATA_DIR / inp
            if cand.exists():
                inp = cand
        if not inp.exists():
            print(f"ERROR: 找不到输入文件: {sys.argv[1]}（data 目录下也不存在）")
            sys.exit(1)
        # 输出：给定路径，或默认与输入同目录同名 .h5ad
        out = Path(sys.argv[2]) if len(sys.argv) >= 3 else inp.with_suffix(".h5ad")
        to_h5ad(inp, out)
        return

    for rel_in, rel_out in CONVERSIONS:
        h5 = DATA_DIR / rel_in
        out = DATA_DIR / rel_out
        if not h5.exists():
            print(f"SKIP (not found): {h5}")
            continue
        to_h5ad(h5, out)


if __name__ == "__main__":
    main()
