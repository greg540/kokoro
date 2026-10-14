@echo off
setlocal
cd /d "%~dp0"
title Local Text + Voice AI Test

if not exist ".venv\Scripts\python.exe" (
    echo Creating a private Python environment for this app...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Python 3.11 launcher entry wasn't found; trying the default Python installation...
        py -3 -m venv .venv
        if errorlevel 1 (
            echo ERROR: Python could not create the virtual environment.
            echo Install 64-bit Python 3.11 from https://www.python.org/downloads/windows/
            echo Ensure the Python Launcher is installed, then run this file again.
            pause
            exit /b 1
        )
    )
)

call .venv\Scripts\activate.bat
if not exist ".venv\deps_installed" (
    python -m pip install --upgrade pip
    if errorlevel 1 goto install_failed
    python -m pip install -r requirements.txt
    if errorlevel 1 goto install_failed
    echo installed>".venv\deps_installed"
)

python app.py
if errorlevel 1 (
    echo.
    echo The app exited with an error. Read the traceback above for details.
    pause
)
exit /b 0

:install_failed
echo.
echo Dependency installation failed. Check the error above and your internet connection.
pause
exit /b 1
