# ============================================================
# nnSVG 方法：Visium Mouse Olfactory Bulb 数据 SVG 检测
# 输入：results/local_results/Visium_Mouse_Olfactory_Bulb/nnSVG/ 下的中间文件
#       counts.mtx (genes x spots), genes.csv, barcodes.csv, location.csv
# 输出：SVG_nnSVG_Visium_Mouse_Olfactory_Bulb.csv + ggplot2 展示图
#       nnSVG_spe.rds (SpatialExperiment 中间对象，便于调试)
# ============================================================
suppressPackageStartupMessages({
  library(nnSVG)
  library(SpatialExperiment)
  library(SingleCellExperiment)
  library(SummarizedExperiment)
  library(scran)
  library(Matrix)
  library(ggplot2)
  library(BiocParallel)
})

# ---------- 路径 ----------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "F:/computatinalbiology/results/local_results/Visium_Mouse_Olfactory_Bulb/nnSVG"
n_threads <- if (length(args) >= 2) as.integer(args[2]) else 14
out_dir <- data_dir

cat("===== nnSVG: 读取中间文件 =====\n")
cat("数据目录:", data_dir, "\n")
cat("线程数:", n_threads, "\n")

counts <- readMM(file.path(data_dir, "counts.mtx"))
genes  <- read.csv(file.path(data_dir, "genes.csv"),  header = FALSE, stringsAsFactors = FALSE)[, 1]
barcodes <- read.csv(file.path(data_dir, "barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
loc_df <- read.csv(file.path(data_dir, "location.csv"), stringsAsFactors = FALSE, row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

# counts 为 dgCMatrix (genes x spots)
counts <- as(counts, "CsparseMatrix")

# 坐标对齐（用像素坐标，与官方教程一致）
loc <- loc_df[barcodes, c("x", "y")]
colnames(loc) <- c("pxl_col_in_fullres", "pxl_row_in_fullres")
rownames(loc) <- barcodes

cat(sprintf("counts 维度: %d genes x %d spots\n", nrow(counts), ncol(counts)))

# ---------- 构建 SpatialExperiment 对象 ----------
spe <- SpatialExperiment(
  assays = list(counts = counts),
  rowData = DataFrame(gene_id = genes, gene_name = genes),
  spatialCoords = as.matrix(loc)
)
rownames(spe) <- genes

# ---------- 过滤基因 ----------
spe <- filter_genes(spe, filter_genes_ncounts = 3,
                    filter_genes_pcspots = 0.5, filter_mito = TRUE)
cat(sprintf("过滤后基因数: %d\n", nrow(spe)))

# ---------- 计算 logcounts ----------
spe <- computeLibraryFactors(spe)
spe <- logNormCounts(spe)
cat("assay names:", paste(assayNames(spe), collapse = ", "), "\n")

# 保存中间对象便于调试
rds_path <- file.path(out_dir, "nnSVG_spe.rds")
saveRDS(spe, rds_path)
cat("已保存中间对象:", rds_path, "\n")

# ---------- 运行 nnSVG ----------
cat("===== 运行 nnSVG =====\n")
set.seed(123)
t0 <- Sys.time()

# Windows 下 MulticoreParam(即 fork) 不可用，nnSVG 内部的默认参数会静默退化为串行。
# 必须显式传入 SnowParam(PSOCK 多进程) 才能真正并行。
bp <- SnowParam(workers = n_threads, type = "SOCK",
                progressbar = FALSE, RNGseed = 123)
cat("使用 BiocParallel 后端:", class(bp)[1], " workers =", bpworkers(bp), "\n")
spe <- nnSVG(spe, BPPARAM = bp, verbose = FALSE)
t1 <- Sys.time()
cat(sprintf("nnSVG 运行耗时: %.2f 分钟\n", as.numeric(difftime(t1, t0, units = "mins"))))

# ---------- 汇总结果 ----------
rd <- as.data.frame(rowData(spe))
res_df <- data.frame(
  gene        = rd$gene_name,
  LR_stat     = rd$LR_stat,
  pval        = rd$pval,
  padj        = rd$padj,
  prop_sv     = rd$prop_sv,
  sigma_sq    = rd$sigma.sq,
  tau_sq      = rd$tau.sq,
  stringsAsFactors = FALSE
)
res_df <- res_df[order(res_df$padj, -res_df$LR_stat, na.last = TRUE), ]
res_df$rank <- seq_len(nrow(res_df))

# ---------- 保存 CSV ----------
csv_path <- file.path(out_dir, "SVG_nnSVG_Visium_Mouse_Olfactory_Bulb.csv")
write.csv(res_df, csv_path, row.names = FALSE)
cat("已保存结果:", csv_path, "\n")
cat(sprintf("显著 SVG (padj < 0.05): %d / %d\n",
            sum(res_df$padj < 0.05, na.rm = TRUE), nrow(res_df)))

# ---------- 展示图 ----------
# 1) Top SVG 的 -log10(padj) 柱状图
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p1 <- ggplot(top_df, aes(x = gene, y = -log10(padj))) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = "nnSVG: Top SVG genes (Visium Mouse Olfactory Bulb)",
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png1 <- file.path(out_dir, "SVG_nnSVG_Visium_Mouse_Olfactory_Bulb_top.png")
ggsave(png1, p1, width = 8, height = 6, dpi = 150)
cat("已保存展示图:", png1, "\n")

# 2) Top-1 SVG 的空间表达图
ix <- which(rowData(spe)$rank == 1)
ix_name <- rowData(spe)$gene_name[ix]
df <- data.frame(
  x = spatialCoords(spe)[, "pxl_col_in_fullres"],
  y = spatialCoords(spe)[, "pxl_row_in_fullres"],
  expr = as.numeric(counts(spe)[ix, ])
)
p2 <- ggplot(df, aes(x = x, y = y, color = expr)) +
  geom_point(size = 0.8) +
  coord_fixed() +
  scale_y_reverse() +
  scale_color_gradient(low = "gray90", high = "blue", trans = "sqrt",
                       breaks = range(df$expr), name = "counts") +
  ggtitle(paste0(ix_name, " (top SVG)")) +
  theme_bw() +
  theme(plot.title = element_text(face = "italic"),
        panel.grid = element_blank(),
        axis.title = element_blank(),
        axis.text = element_blank(),
        axis.ticks = element_blank())

png2 <- file.path(out_dir, "SVG_nnSVG_Visium_Mouse_Olfactory_Bulb_top1_spatial.png")
ggsave(png2, p2, width = 7, height = 6, dpi = 150)
cat("已保存空间表达图:", png2, "\n")

cat("===== nnSVG 完成 ✓ =====\n")
