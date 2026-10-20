@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Local .venv not found. See START_HERE.md to reuse your working environment,
  echo or run scripts\setup_windows.ps1 for an independent installation.
  pause
  exit /b 1
)
if not exist ".env" (
  echo Add your private .env next to app.py first. See .env.example.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8503
pause
