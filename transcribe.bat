@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0transcribe.py" %*
rem Keep the window open when files were dropped onto this script / sent via "Send to".
if not "%~1"=="" pause
