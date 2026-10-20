# Install evaluation dependencies in a separate environment; no API evaluation.
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
if (!(Test-Path '.\.venv-ragas\Scripts\python.exe')) {
    & py -3.11 -m venv .venv-ragas
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 evaluation environment creation failed.' }
}
$python = '.\.venv-ragas\Scripts\python.exe'
& $python -m pip install -r requirements-ragas.txt
if ($LASTEXITCODE -ne 0) { throw 'RAGAS dependency installation failed. Do not install into the app environment.' }
& $python -c "from ragas import evaluate; from ragas.metrics import faithfulness; from langchain_google_genai import ChatGoogleGenerativeAI; print('RAGAS imports OK')"
if ($LASTEXITCODE -ne 0) { throw 'RAGAS import check failed.' }
Write-Host 'No API score has been requested. Use docs\LANGFUSE_RAGAS.md for evaluation commands.'
