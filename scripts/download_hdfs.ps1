# Download HDFS Dataset from Loghub (Zenodo v8)
# Downloads HDFS_1.tar.gz (161.9 MB) which contains both:
#   HDFS_1.log  and  anomaly_label.csv

Write-Host ("=" * 60)
Write-Host "Downloading HDFS Dataset from Loghub / Zenodo"
Write-Host ("=" * 60)
Write-Host ""

$TargetDir = "data\raw\HDFS"
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
}

$TarGzUrl  = "https://zenodo.org/records/8196385/files/HDFS_1.tar.gz?download=1"
$TarGzFile = "$TargetDir\HDFS_1.tar.gz"
$LogFile   = "$TargetDir\HDFS_1.log"
$LabelFile = "$TargetDir\anomaly_label.csv"

# ---- Download archive ----
$alreadyDone = (Test-Path $LogFile) -and (Test-Path $LabelFile)
if ($alreadyDone) {
    Write-Host "HDFS_1.log and anomaly_label.csv already present — skipping download." -ForegroundColor Yellow
}

if (-not $alreadyDone) {
    $hasTar = Test-Path $TarGzFile
    if (-not $hasTar) {
        Write-Host "Downloading HDFS_1.tar.gz (~162 MB) …"
        Invoke-WebRequest -Uri $TarGzUrl -OutFile $TarGzFile
        Write-Host "OK — archive downloaded." -ForegroundColor Green
    } else {
        Write-Host "HDFS_1.tar.gz already present — skipping download." -ForegroundColor Yellow
    }

    # ---- Extract ----
    Write-Host "Extracting HDFS_1.tar.gz …"
    tar -xzf $TarGzFile -C $TargetDir
    Write-Host "OK — extracted." -ForegroundColor Green

    # Some tar layouts nest inside a subfolder; flatten if needed
    $nested = Get-ChildItem $TargetDir -Directory | Select-Object -First 1
    if ($nested -and (Test-Path "$($nested.FullName)\HDFS_1.log")) {
        Move-Item "$($nested.FullName)\*" $TargetDir -Force
        Remove-Item $nested.FullName -ErrorAction SilentlyContinue
    }
}

# ---- Verify ----
Write-Host ""
$missing = @()
if (-not (Test-Path $LogFile))   { $missing += "HDFS_1.log" }
if (-not (Test-Path $LabelFile)) { $missing += "anomaly_label.csv" }

if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing files after extraction:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Contents of $TargetDir :"
    Get-ChildItem $TargetDir | Select-Object Name, Length
    exit 1
}

$logSize = (Get-Item $LogFile).Length / 1MB
$lblSize = (Get-Item $LabelFile).Length / 1KB
Write-Host "  HDFS_1.log       : $([math]::Round($logSize, 1)) MB"
Write-Host "  anomaly_label.csv: $([math]::Round($lblSize, 1)) KB"

Write-Host ""
Write-Host "Done. Run the pipeline next:" -ForegroundColor Green
Write-Host "  python scripts/run_hdfs_pipeline.py" -ForegroundColor Cyan
