# Package SD-MoSE for Publication
# Creates a clean archive excluding development files

$timestamp = Get-Date -Format "yyyy-MM-dd"
$archiveName = "sd-mose-code-$timestamp.zip"

Write-Host "Packaging SD-MoSE for publication..." -ForegroundColor Cyan
Write-Host ""

# Files/folders to include
$include = @(
    "src/",
    "scripts/",
    "tests/",
    "data/README.md",  # Instructions, not actual data
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore"
)

# Create temp directory
$tempDir = "temp_package"
New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

# Copy files
foreach ($item in $include) {
    if (Test-Path $item) {
        $dest = Join-Path $tempDir $item
        $destDir = Split-Path $dest -Parent
        
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        if (Test-Path $item -PathType Container) {
            Copy-Item -Path $item -Destination $dest -Recurse -Force
        } else {
            Copy-Item -Path $item -Destination $dest -Force
        }
        
        Write-Host "✓ Copied: $item" -ForegroundColor Green
    }
}

# Remove Python cache files
Get-ChildItem -Path $tempDir -Recurse -Include "__pycache__","*.pyc","*.pyo" | Remove-Item -Recurse -Force
Write-Host "✓ Cleaned Python cache files" -ForegroundColor Green

# Create ZIP archive
Compress-Archive -Path "$tempDir\*" -DestinationPath $archiveName -Force
Remove-Item -Path $tempDir -Recurse -Force

$size = (Get-Item $archiveName).Length / 1MB

Write-Host ""
Write-Host "✓ Created archive: $archiveName" -ForegroundColor Green
Write-Host "  Size: $($size.ToString('0.00')) MB" -ForegroundColor Yellow
Write-Host ""
Write-Host "Archive contents:" -ForegroundColor Cyan
Write-Host "  - Source code (src/)" -ForegroundColor White
Write-Host "  - Scripts (scripts/)" -ForegroundColor White
Write-Host "  - Tests (tests/)" -ForegroundColor White
Write-Host "  - Documentation (README, CITATION)" -ForegroundColor White
Write-Host "  - Dependencies (requirements.txt)" -ForegroundColor White
Write-Host ""
Write-Host "Excluded:" -ForegroundColor Cyan
Write-Host "  - Virtual environments (.venv)" -ForegroundColor DarkGray
Write-Host "  - Data files (large NetCDF)" -ForegroundColor DarkGray
Write-Host "  - Generated results" -ForegroundColor DarkGray
Write-Host "  - Checkpoints and models" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Ready for publication submission! 🎉" -ForegroundColor Green
