import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from port 8001"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "port": 8001}

if __name__ == "__main__":
    print("Starting test server on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")