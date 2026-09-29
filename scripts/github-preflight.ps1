$ErrorActionPreference = "Stop"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is not installed or is not available in PATH."
}

if (-not (Test-Path ".git")) {
    throw "This directory is not a Git repository. Run `git init` first."
}

Write-Host "Running local verification first..." -ForegroundColor Cyan
& "$PSScriptRoot\verify-local.ps1"

$staged = @(git diff --cached --name-only --diff-filter=ACMRT)

if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the Git index."
}

if ($staged.Count -eq 0) {
    throw "No files are staged. Run `git add .` and then run this preflight again."
}

$forbidden = @(
    '(^|/)\.env$',
    '(^|/)\.venv/',
    '^data/models/(?!\.gitkeep$)',
    '^data/vector_store/',
    '^data/output/(?!\.gitkeep$)',
    '\.(pt|pth|joblib|safetensors)$',
    '^\.stage[^/]*/',
    '^\.v1\.[^/]*/'
)

$violations = @()
foreach ($path in $staged) {
    $normalized = $path -replace '\\', '/'
    foreach ($pattern in $forbidden) {
        if ($normalized -match $pattern) {
            $violations += $path
            break
        }
    }
}

if ($violations.Count -gt 0) {
    Write-Host "Forbidden generated/private files are staged:" -ForegroundColor Red
    $violations | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "Remove these files from the Git index before publishing."
}

$largeFiles = @()
foreach ($path in $staged) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        if ($item.Length -gt 25MB) {
            $largeFiles += "$path ($([math]::Round($item.Length / 1MB, 1)) MB)"
        }
    }
}

if ($largeFiles.Count -gt 0) {
    Write-Host "Unexpected staged files larger than 25 MB:" -ForegroundColor Red
    $largeFiles | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    throw "Large runtime/model artifacts should not be committed to this portfolio repository."
}

Write-Host "GitHub preflight passed." -ForegroundColor Green
Write-Host "Review `git status` once more before committing." -ForegroundColor Yellow
