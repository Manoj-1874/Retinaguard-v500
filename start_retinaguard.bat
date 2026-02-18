@echo off
echo ================================================================================
echo RETINAGUARD V500 - SYSTEM STARTUP
echo ================================================================================
echo.
echo This will start all required services for RetinaGuard V500
echo.
echo Press Ctrl+C to stop all servers
echo ================================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    pause
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js not found
    pause
    exit /b 1
)

echo [1/3] Starting Flask AI Server (Port 5001)...
start "Flask AI Server" cmd /k "python app.py"
timeout /t 3 >nul

echo [2/3] Starting Node.js + MongoDB Server (Port 5000)...
start "Node.js Server" cmd /k "node server.js"
timeout /t 2 >nul

echo [3/3] Opening Browser...
timeout /t 2 >nul
start http://localhost:5000

echo.
echo ================================================================================
echo ALL SYSTEMS ONLINE!
echo ================================================================================
echo.
echo   Flask AI Server:  http://localhost:5001
echo   Web Application:  http://localhost:5000
echo.
echo   Press any key to view system status...
echo ================================================================================
pause

curl http://localhost:5001/api/health
echo.
echo ================================================================================
echo System is running. Close this window to keep servers active.
echo To stop servers: Close the Flask and Node.js terminal windows.
echo ================================================================================
pause
