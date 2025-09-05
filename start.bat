@echo off
echo =====================================================
echo Agricultural AI Platform - Windows Startup Script
echo =====================================================

echo Checking Node.js installation...
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo Checking MongoDB installation...
mongod --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: MongoDB is not installed or not in PATH. Please install MongoDB Community Edition.
    echo You can download it from https://www.mongodb.com/try/download/community
)

echo.
echo Starting Agricultural AI Platform...
echo.

echo [1/4] Installing backend dependencies...
cd backend
if not exist node_modules (
    npm install
    if %errorlevel% neq 0 (
        echo Error: Failed to install backend dependencies
        pause
        exit /b 1
    )
)

echo [2/4] Installing frontend dependencies...
cd ..\frontend
if not exist node_modules (
    npm install
    if %errorlevel% neq 0 (
        echo Error: Failed to install frontend dependencies
        pause
        exit /b 1
    )
)

echo [3/4] Starting MongoDB (if not already running)...
net start MongoDB >nul 2>&1

echo [4/4] Starting application services...
cd ..

echo.
echo Starting backend server...
start "Backend Server" cmd /k "cd backend && npm start"

timeout /t 3 /nobreak >nul

echo Starting frontend server...
start "Frontend Server" cmd /k "cd frontend && npm start"

echo.
echo =====================================================
echo Agricultural AI Platform is starting up!
echo =====================================================
echo Backend API: http://localhost:3000
echo Frontend Dashboard: http://localhost:8080
echo.
echo The application will open in your default browser shortly.
echo Press any key to close this window...
echo =====================================================

timeout /t 5 /nobreak >nul
start http://localhost:8080

pause >nul
