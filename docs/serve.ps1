# PowerShell script to build and serve Sphinx documentation

Write-Host "Building Sphinx documentation..." -ForegroundColor Cyan
cd $PSScriptRoot
uv run sphinx-build -b html . _build/html

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nDocumentation built successfully!" -ForegroundColor Green
    Write-Host "Starting local server on http://localhost:8000" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Gray
    
    cd _build/html
    python -m http.server 8000
} else {
    Write-Host "Build failed. Please check the errors above." -ForegroundColor Red
    exit 1
}
