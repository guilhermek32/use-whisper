@echo off
rem uv installs/syncs the environment on first run; falls back to an existing .venv.
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run --project "%~dp0" "%~dp0transcribe.py" %*
) else (
    "%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" %*
)
rem Keep the window open when files were dropped onto this script / sent via "Send to".
if not "%~1"=="" pause
