# =============================================================================
# run_benchmark.ps1 -- SVG method benchmark pipeline (PowerShell)
# =============================================================================
# Steps:
#   1) Preprocessing:        src/preprocess/h5ad_preprocess.py
#   2) Run methods:          SPARK-X, nnSVG, SpaGCN, SpaSEG
#   3) Evaluation:           src/utils/evaluation.py
#
# Environments:
#   - Python: env_spatial/python.exe (or SVG_PYTHON env var)
#   - R:      env_R/lib/R/bin/Rscript.exe (conda R 4.4.3, renv auto-activated)
#             (or SVG_RSCRIPT env var to override)
#
# Usage:
#   .\src\pipeline\run_benchmark.ps1
#   .\src\pipeline\run_benchmark.ps1 -dataset mouse_brain_STARmap
#   .\src\pipeline\run_benchmark.ps1 -dataset mouse_brain_STARmap -methods spark,nnsvg
# =============================================================================
param(
  [string]$dataset = "mouse_brain_STARmap",
  [string]$h5ad = "",
  [string]$spatial = "",
  [string]$outdir = "",
  [string]$sample = "",
  [string]$methods = "spark,nnsvg,spagcn,spaseg",
  [string]$device = "auto",
  [switch]$skip_preprocess,
  [switch]$skip_eval
)

$ROOT = "f:\computatinalbiology"
$PYTHON = if ($env:SVG_PYTHON) { $env:SVG_PYTHON } else { "$ROOT\env_spatial\python.exe" }
$RSCRIPT = if ($env:SVG_RSCRIPT) { $env:SVG_RSCRIPT } else { "$ROOT\env_R\lib\R\bin\Rscript.exe" }

function run_R {
  param([string]$script, [string]$method_dir, [string]$sample_name, [string]$log_file)
  $old_path = $env:Path
  $env:Path = "$ROOT\env_R\Library\bin;$ROOT\env_R\bin;$ROOT\env_R\lib\R\bin;$env:Path"
  & $RSCRIPT $script $method_dir $sample_name 2>&1 | Tee-Object -FilePath $log_file
  $env:Path = $old_path
}

$ts = { Get-Date -Format "HH:mm:ss" }
function log_header($title) {
  "`n============================================================"
  "  [$(&$ts)] $title"
  "============================================================"
}
function log_msg($msg) { "  [$(&$ts)] $msg" }

log_header "run_benchmark: SVG method benchmark"
log_msg "ROOT = $ROOT"
log_msg "PYTHON = $PYTHON"
log_msg "RSCRIPT = $RSCRIPT"
log_msg "DATASET = $dataset"

$methods_list = $methods.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }

try {
  $cmd = "import sys; sys.path.insert(0, r'$ROOT'); import src; print(src.resolve_run(dataset=r'$dataset')['dim'])"
  $dim = (& $PYTHON -c $cmd) | Select-Object -Last 1
  if ("$dim".Trim() -eq "3") {
    Write-Host "dim=3: only SPARK-X supports 3D, skipping nnSVG/SpaGCN/SpaSEG"
    $methods_list = @($methods_list | Where-Object { $_ -eq 'spark' })
    if ($methods_list.Count -eq 0) { Write-Error "dim=3 no available methods (only spark)"; exit 1 }
  }
}
catch { }

$outdir = if ($outdir) { $outdir } else { "$ROOT\results\local_results\$dataset" }
$sample = if ($sample) { $sample } else { "STARmap_Mouse_Brain" }

New-Item -ItemType Directory -Force -Path "$outdir\logs" | Out-Null
foreach ($m in $methods_list) {
  $subdir = @{ "spark"="SPARK_X"; "nnsvg"="nnSVG"; "spagcn"="spaGCN"; "spaseg"="spaSEG" }[$m]
  New-Item -ItemType Directory -Force -Path "$outdir\$subdir" | Out-Null
}

if (-not $skip_preprocess) {
  log_header "Step 1/3: Preprocessing"
  & $PYTHON "$ROOT\src\preprocess\h5ad_preprocess.py" --dataset $dataset --outdir $outdir --sample-name $sample --methods $methods_list 2>&1 | Tee-Object -FilePath "$outdir\logs\preprocess.log"
  if ($LASTEXITCODE -ne 0) { Write-Error "Preprocessing failed (exit $LASTEXITCODE)"; exit 1 }
  log_msg "Preprocessing done"
}

log_header "Step 2/3: Running methods"
foreach ($m in $methods_list) {
  $subdir = @{ "spark"="SPARK_X"; "nnsvg"="nnSVG"; "spagcn"="spaGCN"; "spaseg"="spaSEG" }[$m]
  $method_dir = "$outdir\$subdir"

  if ($m -in @("spark", "nnsvg")) {
    log_header ("${m}: ${sample}")
    $t0 = Get-Date
    $script = "$ROOT\src\r_models\run_${m}.r"
    $log_path = "$outdir\logs\${m}.log"
    run_R -script $script -method_dir $method_dir -sample_name $sample -log_file $log_path
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    log_msg "${m} done ($($elapsed)s)"
  } elseif ($m -in @("spagcn", "spaseg")) {
    log_header ("${m}: ${sample}")
    $t0 = Get-Date
    $log_path = "$outdir\logs\${m}.log"
    & $PYTHON "$ROOT\src\py_models\run_${m}.py" --dataset $dataset --outdir $outdir --sample $sample --device $device 2>&1 | Tee-Object -FilePath $log_path
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    log_msg "${m} done ($($elapsed)s)"
  }

  if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne 3221225781) {
    Write-Warning "${m} returned non-zero exit code ($LASTEXITCODE)"
  }
}

if (-not $skip_eval) {
  log_header "Step 3/3: Evaluation"
  & $PYTHON "$ROOT\src\utils\evaluation.py" --dataset $dataset --outdir $outdir --sample $sample --methods $methods 2>&1 | Tee-Object -FilePath "$outdir\logs\evaluation.log"
  if ($LASTEXITCODE -ne 0) { log_msg "[WARNING] evaluation.py returned non-zero" }
}

log_header "run_benchmark: ALL DONE"
log_msg "Output: $outdir"
log_msg "Logs:   $outdir\logs"
