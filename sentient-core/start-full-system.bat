@echo off
echo ========================================
echo  Sentient Core - Full System Startup
echo ========================================
echo.

REM Change to the project directory
cd /d "d:\sentientcore\sentient-core"

echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)
echo Python found!

echo.
echo [2/6] Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 18+ and add it to your PATH
    pause
    exit /b 1
)
echo Node.js found!

echo.
echo [3/6] Installing Python dependencies...
echo Installing backend requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)
echo Backend dependencies installed successfully!

echo.
echo [4/6] Installing Frontend dependencies...
cd frontend
npm install
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)
echo Frontend dependencies installed successfully!
cd ..

echo.
echo [5/6] Verifying environment files...
if not exist ".env" (
    echo WARNING: .env file not found in backend
    echo Creating .env from .env.example...
    copy ".env.example" ".env"
    echo Please edit .env file with your API keys before running again
    pause
    exit /b 1
)
echo Backend .env file found!

if not exist "frontend\.env.local" (
    echo WARNING: .env.local file not found in frontend
    echo Creating frontend/.env.local...
    echo NEXT_PUBLIC_API_URL=http://localhost:8000 > "frontend\.env.local"
)
echo Frontend .env.local file found!

echo.
echo [6/6] Starting servers...
echo.
echo Starting Backend Server (FastAPI on port 8000)...
start "Backend Server" cmd /k "cd /d d:\sentientcore\sentient-core && uvicorn app.api.app:app --host 0.0.0.0 --port 8000 --reload"

echo Waiting 5 seconds for backend to start...
timeout /t 5 /nobreak >nul

echo Starting Frontend Server (Next.js on port 3000)...
start "Frontend Server" cmd /k "cd /d d:\sentientcore\sentient-core\frontend && npm run dev"

echo.
echo ========================================
echo  System Startup Complete!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo Frontend App: http://localhost:3000
echo API Docs: http://localhost:8000/docs
echo.
echo Both servers are starting in separate windows.
echo Wait a few moments for them to fully initialize.
echo.
echo To test Build 1 & 2 implementations:
echo 1. Open http://localhost:3000 in your browser
echo 2. Try basic conversation (Build 1)
echo 3. Try research requests (Build 2)
echo.
echo Press any key to open the frontend in your default browser...
pause >nul
start http://localhost:3000

echo.
echo Batch file complete. Servers are running in separate windows.
echo Close those windows to stop the servers.
pause