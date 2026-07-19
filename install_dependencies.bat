@echo off
echo ===================================================
echo Installing Dependencies for RetinaGuard V500
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo Python found. Installing Python dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install -r requirements.txt

echo.
echo Installing Node.js dependencies...
echo.

REM Check if npm is installed
npm --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: npm is not installed or not in PATH. Skipping Node.js dependencies.
    echo Please install Node.js if you need to run the Node server.
) else (
    echo npm found. Installing Node.js dependencies...
    call npm install
)

echo.
echo ===================================================
echo Installation Complete!
echo ===================================================
echo.
echo Next steps:
echo 1. Start the Node.js Server: node server.js
echo 2. Start the Flask AI Server: python app.py
echo.
pause
