@echo off
echo ==========================================
echo   STARTING MONGODB SERVICE
echo ==========================================
echo.

net start MongoDB

if %errorlevel% == 0 (
    echo.
    echo ✓ MongoDB started successfully!
    echo.
    echo MongoDB is now running on: mongodb://127.0.0.1:27017
    echo.
    echo You can close this window and restart the Node server.
) else (
    echo.
    echo × Failed to start MongoDB
    echo.
    echo Make sure you ran this file as Administrator:
    echo   Right-click start_mongodb.bat ^> Run as Administrator
)

echo.
echo Press any key to close...
pause > nul
