#!/usr/bin/env python3
"""
Development Server with Auto-Reload

This script runs the Sentient Core API with uvicorn's auto-reload functionality.
Note: There is a known issue where the server may hang during startup when reload=True
is enabled, particularly related to the SentenceTransformer model loading in the research router.

If the server hangs during startup:
1. Stop the process (Ctrl+C)
2. Use main.py instead (which runs without reload)
3. Manually restart the server when you make changes

Alternatively, you can try:
- Running with a smaller timeout
- Disabling specific routers temporarily
- Using external tools like nodemon to watch for file changes
"""

import uvicorn
import sys
import os
from dotenv import load_dotenv

def main():
    print("🚀 Starting Sentient Core Development Server with Auto-Reload")
    print("⚠️  Warning: Server may hang during startup due to model loading.")
    print("   If it hangs, use Ctrl+C and run main.py instead.\n")
    
    # Load environment variables
    load_dotenv()
    
    try:
        uvicorn.run(
            "app.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            reload_dirs=["app", "core"],  # Only watch specific directories
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Development server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\n💡 Try running main.py instead for a stable server without auto-reload")
        sys.exit(1)

if __name__ == "__main__":
    main()