from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Simple server is running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "API is working"}

@app.get("/test")
async def test():
    return {"message": "Test endpoint", "data": {"working": True}}

if __name__ == "__main__":
    print("Starting simple server on port 8002...")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")