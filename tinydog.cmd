@echo off
setlocal
set "BOOTSTRAP=%~dp0scripts\tinydog.py"

where py >nul 2>nul
if not errorlevel 1 goto run_py

where python >nul 2>nul
if not errorlevel 1 goto run_python

echo Python 3.10 or newer is required. Install Python, then run this command again. 1>&2
exit /b 1

:run_py
py -3 "%BOOTSTRAP%" %*
exit /b

:run_python
python "%BOOTSTRAP%" %*
exit /b
