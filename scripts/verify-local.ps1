$ErrorActionPreference = "Stop"

Write-Host "Checking installed package version..." -ForegroundColor Cyan
$installedVersion = python -c "import importlib.metadata as m; print(m.version('opsmind-ai'))"
if ($installedVersion.Trim() -ne "1.0.2") {
    throw "Installed opsmind-ai version is $installedVersion. Run: python -m pip install -e `".[dev]`""
}

Write-Host "Running dependency check..." -ForegroundColor Cyan
python -m pip check

Write-Host "Compiling source..." -ForegroundColor Cyan
python -m compileall -q src

Write-Host "Running test suite..." -ForegroundColor Cyan
python -m pytest -q

Write-Host "Local verification passed for OpsMind v1.0.2." -ForegroundColor Green
