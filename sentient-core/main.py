if __name__ == "__main__":
    import uvicorn
    # Note: reload=False to avoid hanging issues with the current codebase
    # For development with auto-reload, use dev_server.py instead
    uvicorn.run("app.api.app:app", host="0.0.0.0", port=8000, reload=False)