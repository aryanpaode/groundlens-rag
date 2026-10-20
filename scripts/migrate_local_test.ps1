# Compatibility wrapper: copies only private LOCAL STATE, never .env or cloud vectors.
param([Parameter(Mandatory=$true)][string]$Source)
$ErrorActionPreference = 'Stop'
& py -3.11 (Join-Path $PSScriptRoot 'restore_local_state.py') --source $Source
if ($LASTEXITCODE -ne 0) { throw 'State restoration failed. Read the message above.' }
Write-Host 'Add your .env separately. See START_HERE.md.'
