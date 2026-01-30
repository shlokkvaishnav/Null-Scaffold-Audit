# Cleanup script - Remove temporary files and old results
# Usage: .\cleanup.ps1

Write-Host "🧹 Cleaning up temporary files and old results..." -ForegroundColor Cyan

# Remove log files
Write-Host "`nRemoving log files..." -ForegroundColor Yellow
Remove-Item -Path "*.log" -Force -ErrorAction SilentlyContinue

# Remove PySR temporary files
Write-Host "Removing PySR temp files..." -ForegroundColor Yellow
if (Test-Path "pysr_tmp") {
    Remove-Item -Path "pysr_tmp" -Recurse -Force
}

# Clean checkpoints (keep final only)
Write-Host "Cleaning checkpoints (keeping final models)..." -ForegroundColor Yellow
Remove-Item -Path "checkpoints\sdmose_iter*.pth" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "checkpoints\ensemble" -Recurse -Force -ErrorAction SilentlyContinue

# Clean old results
Write-Host "Cleaning old results..." -ForegroundColor Yellow
Remove-Item -Path "results\equations_iter*.txt" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "results\ablation_*.csv" -Force -ErrorAction SilentlyContinue

# Clean Python cache
Write-Host "Cleaning Python cache..." -ForegroundColor Yellow
Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force

# Clean pytest cache
if (Test-Path ".pytest_cache") {
    Remove-Item -Path ".pytest_cache" -Recurse -Force
}

Write-Host "`n✅ Cleanup complete!" -ForegroundColor Green
Write-Host "`nKept:" -ForegroundColor Cyan
Write-Host "  - Final model checkpoints"
Write-Host "  - Latest equations"
Write-Host "  - Final figures"
Write-Host "  - Source code & tests"
Write-Host "  - Documentation (README.md, ULTIMATE_GUIDE.md)"
