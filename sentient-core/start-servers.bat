@echo off
echo ========================================
echo  Sentient Core - Quick Server Startup
echo ========================================
echo.

REM Change to the project directory
cd /d "d:\sentientcore\sentient-core"

echo Checking environment files...
if not exist ".env" (
    echo ERROR: .env file not found!
    echo Please run start-full-system.bat first to set up the environment
    pause
    exit /b 1
)

if not exist "frontend\.env.local" (
    echo Creating frontend/.env.local...
    echo NEXT_PUBLIC_API_URL=http://localhost:8000 > "frontend\.env.local"
)

echo.
echo Starting Backend Server (FastAPI on port 8000)...
start "Backend Server" cmd /k "cd /d d:\sentientcore\sentient-core && uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Frontend Server (Next.js on port 3000)...
start "Frontend Server" cmd /k "cd /d d:\sentientcore\sentient-core\frontend && npm run dev"

echo.
echo ========================================
echo  Quick Startup Complete!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend App: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Both servers are starting in separate windows.
echo.
echo Press any key to open the frontend...
pause >nul
start http://localhost:3000

echo.
echo Servers are running. Close the server windows to stop them.
pause