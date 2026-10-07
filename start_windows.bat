@echo off
rem Double-click to start TraceRAG. The first run creates a private Python
rem environment in .venv and installs the packages (a few minutes).
setlocal
cd /d "%~dp0"

if exist ".venv\installed.ok" goto run

rem Find Python 3.10 to 3.14 (PyTorch has no packages for newer versions yet).
set "PY="
for %%V in (3.13 3.12 3.14 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 14) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY goto nopython

echo Setting up TraceRAG for the first time with %PY%. This takes a few minutes.
%PY% -m venv --clear .venv
if errorlevel 1 goto nopython
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo ok> ".venv\installed.ok"

:run
".venv\Scripts\python.exe" run_app.py %*
goto end

:nopython
echo.
echo TraceRAG needs Python 3.10 to 3.14 (3.13 is a good choice).
echo Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH"
echo during installation, then run this file again.
goto end

:failed
echo.
echo Installing the packages failed. Read the messages above, fix the problem, then run this file again.

:end
pause
