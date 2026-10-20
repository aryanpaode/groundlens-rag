# Creates an independent main-app venv. Does not modify an older installation.
param([switch]$Advanced)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (!(Test-Path '.\.venv\Scripts\python.exe')) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 environment creation failed.' }
}
$python = '.\.venv\Scripts\python.exe'
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Base dependency installation failed.' }
if ($Advanced) {
    & $python -m pip install -r requirements-advanced.txt
    if ($LASTEXITCODE -ne 0) { throw 'Optional dependency installation failed.' }
}
if (!(Test-Path '.env')) { Copy-Item '.env.example' '.env' }
Write-Host 'Main environment ready. Edit private .env and follow START_HERE.md.'
Write-Host 'RAGAS is separate: scripts\setup_ragas.ps1. No cloud requests have been made by this script.'
