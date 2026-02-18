@echo off
echo ========================================
echo Installing Python Dependencies for V500
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9 or higher from python.org
    pause
    exit /b 1
)

echo Python found. Installing dependencies...
echo.

REM Upgrade pip first
python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install -r requirements.txt

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Place your trained models in the 'models' folder
echo 2. Run: python app.py
echo 3. Flask server will start on http://localhost:5001
echo.
pause
