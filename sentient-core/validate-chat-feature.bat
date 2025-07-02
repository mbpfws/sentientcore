@echo off
echo Sentient Core Chat Feature Validation
echo ====================================
echo.
echo This script will help you validate the chat feature end-to-end.
echo.

REM Check if servers are running
powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:8000/health' -UseBasicParsing } catch { $null }; if ($response -and $response.StatusCode -eq 200) { exit 0 } else { exit 1 }"

if %ERRORLEVEL% NEQ 0 (
  echo Backend server is not running. Please start the servers first using 'start-dev.bat'
  exit /b 1
)

powershell -Command "$response = try { Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing } catch { $null }; if ($response -and $response.StatusCode -eq 200) { exit 0 } else { exit 1 }"

if %ERRORLEVEL% NEQ 0 (
  echo Frontend server is not running. Please start the servers first using 'start-dev.bat'
  exit /b 1
)

echo Both servers are running!
echo.
echo Running backend unit tests...
cd app
python -m pytest tests/test_chat_api.py -v
cd ..

echo.
echo ====================================
echo.
echo To validate the chat feature end-to-end:
echo.
echo 1. Open your browser and navigate to http://localhost:3000
echo 2. Select a workflow mode (Intelligent RAG, Multi-Agent RAG, etc.)
echo 3. Type a message in the chat box and click Send
echo 4. Verify that the message is sent to the backend and a response is received
echo 5. Try different research modes and verify they work correctly
echo.
echo API documentation is available at http://localhost:8000/docs
echo.

echo Would you like to open the application in your browser now? (Y/N)
set /p choice="> "

if /i "%choice%"=="Y" (
  start http://localhost:3000
  start http://localhost:8000/docs
)
