# ============================================================
# SPARK-X 方法：mouse_brain_STARmap 数据 SVG 检测
# 输入：results/local_results/mouse_brain_STARmap/SPARK_X/ 下的中间文件
#       counts.mtx (genes x spots), genes.csv, barcodes.csv, location.csv
# 输出：SVG_SPARK_mouse_brain_STARmap.csv + ggplot2 展示图
# ============================================================
suppressPackageStartupMessages({
  library(SPARK)
  library(Matrix)
  library(ggplot2)
})

# ---------- 路径 ----------
args <- commandArgs(trailingOnly = TRUE)
data_dir <- if (length(args) >= 1) args[1] else
  "F:/computatinalbiology/results/local_results/mouse_brain_STARmap/SPARK_X"
out_dir <- data_dir

cat("===== SPARK-X: 读取中间文件 =====\n")
cat("数据目录:", data_dir, "\n")

counts <- readMM(file.path(data_dir, "counts.mtx"))
genes  <- read.csv(file.path(data_dir, "genes.csv"),  header = FALSE, stringsAsFactors = FALSE)[, 1]
barcodes <- read.csv(file.path(data_dir, "barcodes.csv"), header = FALSE, stringsAsFactors = FALSE)[, 1]
loc_df <- read.csv(file.path(data_dir, "location.csv"), stringsAsFactors = FALSE, row.names = 1)

rownames(counts) <- genes
colnames(counts) <- barcodes

# counts 为 dgCMatrix (genes x spots)，转成 sparseMatrix 供 sparkx
counts <- as(counts, "sparseMatrix")

# 坐标对齐：location.csv 行名即 barcode
loc <- as.matrix(loc_df[barcodes, c("x", "y")])
rownames(loc) <- barcodes

cat(sprintf("counts 维度: %d genes x %d spots\n", nrow(counts), ncol(counts)))
cat(sprintf("location 维度: %d x %d\n", nrow(loc), ncol(loc)))

# ---------- 去除线粒体基因 ----------
mt_idx <- grep("^mt-", rownames(counts), ignore.case = TRUE)
if (length(mt_idx) != 0) {
  counts <- counts[-mt_idx, ]
  cat(sprintf("去除线粒体基因 %d 个\n", length(mt_idx)))
}

# ---------- 运行 SPARK-X ----------
cat("===== 运行 sparkx (mixture) =====\n")
res <- sparkx(counts, loc, numCores = 1, option = "mixture", verbose = FALSE)

# ---------- 汇总结果 ----------
res_mtest <- res$res_mtest
res_df <- data.frame(
  gene        = rownames(res_mtest),
  combinedPval = res_mtest$combinedPval,
  adjustedPval = res_mtest$adjustedPval,
  stringsAsFactors = FALSE
)
# 投影核统计量与 p 值（用于参考，sparkx.sk 的 stat 为投影核的统计量）
if ("projection" %in% colnames(res$stats)) {
  res_df$projection_stat <- as.numeric(res$stats[res_df$gene, "projection"])
  res_df$projection_pval <- as.numeric(res$res_stest[res_df$gene, "projection"])
}

# 排名（按 adjustedPval 升序，最小为 1）
res_df <- res_df[order(res_df$adjustedPval, res_df$combinedPval, na.last = TRUE), ]
res_df$rank <- seq_len(nrow(res_df))

# ---------- 保存 CSV ----------
csv_path <- file.path(out_dir, "SVG_SPARK_mouse_brain_STARmap.csv")
write.csv(res_df, csv_path, row.names = FALSE)
cat("已保存结果:", csv_path, "\n")
cat(sprintf("显著 SVG (adjustedPval < 0.05): %d / %d\n",
            sum(res_df$adjustedPval < 0.05, na.rm = TRUE), nrow(res_df)))

# ---------- 展示图 ----------
top_n <- 30
top_df <- head(res_df, top_n)
top_df$gene <- factor(top_df$gene, levels = rev(top_df$gene))

p <- ggplot(top_df, aes(x = gene, y = -log10(adjustedPval))) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "SPARK-X: Top SVG genes (mouse_brain_STARmap)",
       x = "Gene", y = "-log10(adjusted P-value)") +
  theme_minimal(base_size = 11) +
  theme(plot.title = element_text(hjust = 0.5))

png_path <- file.path(out_dir, "SVG_SPARK_mouse_brain_STARmap_top.png")
ggsave(png_path, p, width = 8, height = 6, dpi = 150)
cat("已保存展示图:", png_path, "\n")

cat("===== SPARK-X 完成 ✓ =====\n")
