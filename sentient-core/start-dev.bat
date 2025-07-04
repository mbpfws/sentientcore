@echo off
echo Starting Sentient Core Development Environment
echo =======================================

start cmd /k "cd frontend && npm run dev"
start cmd /k "python -m uvicorn app.api.app:app --reload --port 8000"

echo Development servers started!
echo Frontend: http://localhost:3000
echo Backend: http://localhost:8000
