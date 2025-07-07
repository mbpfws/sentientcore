@echo off
echo ========================================
echo   SentientCore Development Server
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "development_api_server.py" (
    echo ERROR: development_api_server.py not found!
    echo Please run this script from the sentient-core directory.
    echo Current directory: %CD%
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please ensure .venv directory exists with Python virtual environment.
    echo You may need to run: python -m venv .venv
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Checking Python and dependencies...
python --version
echo.

REM Check if FastAPI is installed
python -c "import fastapi; print('FastAPI version:', fastapi.__version__)" 2>nul
if errorlevel 1 (
    echo WARNING: FastAPI not found. Installing dependencies...
    pip install fastapi uvicorn requests pydantic
    echo.
)

echo Starting development server...
echo.
echo Server will be available at:
echo   - Main: http://localhost:8007
echo   - Docs: http://localhost:8007/docs
echo   - ReDoc: http://localhost:8007/redoc
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

REM Start the server
python development_api_server.py

echo.
echo Server stopped.
pause