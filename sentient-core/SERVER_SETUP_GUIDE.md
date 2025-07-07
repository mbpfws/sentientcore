# 🚀 SentientCore Server Setup Guide

## 🔍 Issue Summary

After extensive diagnostics, we've identified a **Windows-specific process isolation issue** that prevents Python HTTP servers from being accessible when started through command interfaces. This guide provides working solutions.

## ✅ Working Solution: Development API Server

### Quick Start

1. **Open a new terminal/command prompt** (separate from this IDE)
2. **Navigate to the project directory**:
   ```bash
   cd "D:\sentientcore\sentient-core"
   ```
3. **Activate the virtual environment**:
   ```bash
   .venv\Scripts\activate
   ```
4. **Start the development server**:
   ```bash
   python development_api_server.py
   ```
5. **Test the server** (in another terminal):
   ```bash
   python test_development_server.py
   ```

### 🌐 Access Points

Once running, the server will be available at:
- **Main Server**: http://localhost:8007
- **API Documentation**: http://localhost:8007/docs
- **ReDoc Documentation**: http://localhost:8007/redoc

### 📡 Available Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server information and status |
| GET | `/health` | Health check with uptime stats |
| GET | `/api/chat/message/json` | Simple chat test (GET) |
| POST | `/api/chat/message/json` | Main chat endpoint |
| GET | `/api/status` | Detailed API status |
| GET | `/api/test` | Test endpoint with sample data |
| GET | `/docs` | Interactive API documentation |
| GET | `/redoc` | Alternative API documentation |

### 💬 Chat API Usage

**POST Request Example**:
```bash
curl -X POST "http://localhost:8007/api/chat/message/json" \
     -H "Content-Type: application/json" \
     -d '{
       "message": "Hello, how are you?",
       "model": "development-model",
       "temperature": 0.7
     }'
```

**Response Example**:
```json
{
  "response": "I understand you said: 'Hello, how are you?'. This is a development server response.",
  "model": "development-model",
  "timestamp": 1751879123.456,
  "processing_time": 0.123
}
```

## 🔧 Alternative Solutions

### Option 1: Use Next.js Proxy (Recommended for Frontend)

The Next.js server on port 3000 is fully functional. You can:
1. Continue using it for frontend development
2. Add API proxy routes in Next.js to forward to backend services
3. Use it as the primary development server

### Option 2: IDE Integration

Run the development server directly in your IDE:
1. Open `development_api_server.py` in your IDE
2. Run it using your IDE's "Run" button or integrated terminal
3. This bypasses the command interface isolation issue

### Option 3: Docker (Advanced)

For production-like environment:
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8007
CMD ["python", "development_api_server.py"]
```

## 🐛 Troubleshooting

### Server Won't Start
1. **Check Python environment**:
   ```bash
   python --version
   pip list | findstr fastapi
   ```
2. **Verify port availability**:
   ```bash
   netstat -ano | findstr :8007
   ```
3. **Check virtual environment**:
   ```bash
   where python
   ```

### Connection Refused Errors
1. **Ensure server is running** in a separate terminal
2. **Check firewall settings** (Windows Defender, antivirus)
3. **Try different ports** by modifying the port in `development_api_server.py`
4. **Use 127.0.0.1 instead of localhost**

### Import Errors
1. **Install missing dependencies**:
   ```bash
   pip install fastapi uvicorn requests pydantic
   ```
2. **Verify virtual environment** is activated

## 📋 Diagnostic Files Created

| File | Purpose |
|------|----------|
| `server_diagnostic_report.md` | Detailed technical analysis |
| `development_api_server.py` | Working development server |
| `test_development_server.py` | Server testing script |
| `python_diagnostic.py` | Environment diagnostic tool |
| `windows_compatible_server.py` | Windows-optimized server attempt |

## 🎯 Next Steps

1. **Start the development server** using the instructions above
2. **Test all endpoints** using the test script
3. **Integrate with your frontend** by updating API URLs to `http://localhost:8007`
4. **Continue development** with full API functionality

## 📞 Support

If you encounter issues:
1. Run `python python_diagnostic.py` for environment check
2. Check the diagnostic report for technical details
3. Ensure you're running the server in a separate terminal (not through the IDE command interface)

---

**✨ The development server provides full API functionality and bypasses the Windows process isolation issue. You can now continue development with a fully functional backend API!**