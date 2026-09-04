# 项目级 R 配置：直接挂载 renv 项目库，使 R 方法（SPARK-X / nnSVG）可 library() 到全部依赖。
#
# 说明：不用 renv::load()/activate.R 自动激活 —— 那会在每次 R 启动时联网
# （默认 CRAN/cloud）做仓库校验，可能阻塞或失败；本项目的环境一致性已由
# setup_linux_env.sh（conda 前缀 + renv::restore）保证，运行期只需把项目库挂上。
# 项目库布局：renv/library/R-<major.minor>/<platform>
libs <- Sys.glob(file.path("renv", "library",
                           paste0("R-", as.integer(R.version$major), ".",
                                  as.integer(R.version$minor)), "*"))
if (length(libs)) .libPaths(c(libs, .libPaths()))
