# PowerShell script to build Sphinx documentation

Write-Host "Building Sphinx documentation..." -ForegroundColor Cyan
cd $PSScriptRoot
uv run sphinx-build -b html . _build/html

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDocumentation built successfully!" -ForegroundColor Green
    Write-Host "Location: docs/_build/html/index.html" -ForegroundColor Cyan
} else {
    Write-Host "Build failed. Please check the errors above." -ForegroundColor Red
    exit 1
}
