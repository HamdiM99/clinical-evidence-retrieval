# Week 1 on Windows (PowerShell), equivalent of: make setup test up download ingest eval report
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File scripts\run_week1.ps1
param([string[]]$Datasets = @("nfcorpus", "scifact"))
$ErrorActionPreference = "Stop"

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Python virtual environment"
if (-not (Test-Path .venv)) {
    # first Python >= 3.10 found by the py launcher
    $installed = (py --list) -join " "
    $ver = @("3.12", "3.11", "3.10") | Where-Object { $installed -match "-V:$([regex]::Escape($_))\b" } | Select-Object -First 1
    if (-not $ver) { throw "Python 3.10+ not found (py launcher)" }
    py "-$ver" -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[embed,dev]"

Step "Unit and end-to-end tests (no Elasticsearch)"
pytest -q

Step "Elasticsearch (Docker Desktop)"
docker compose up -d elasticsearch
Write-Host "waiting for Elasticsearch..."
for ($i = 0; $i -lt 60; $i++) {
    try { $h = Invoke-RestMethod "http://localhost:9200/_cluster/health"; if ($h.status -ne "red") { break } } catch {}
    Start-Sleep -Seconds 3
}
Invoke-RestMethod "http://localhost:9200" | Select-Object -ExpandProperty version | Select-Object number

Step "Integration tests"
$env:ES_URL = "http://localhost:9200"
pytest -q -m integration
Remove-Item Env:ES_URL

foreach ($d in $Datasets) {
    Step "Dataset $d : download, ingest, evaluate"
    cer download $d
    cer ingest "configs/${d}_hybrid.yaml"
    cer eval "configs/${d}_bm25.yaml" "configs/${d}_dense.yaml" "configs/${d}_hybrid.yaml"
}

Step "Report"
cer report
Write-Host "`nDone. Results: reports\results.md, details: runs\<name>\" -ForegroundColor Green
