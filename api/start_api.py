#!/usr/bin/env python3
"""
SMB Research Agent API Startup Script
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    print("⚠️  Warning: python-dotenv not found. Make sure to set GOOGLE_API_KEY manually.")

# Import and run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    from main import app
    
    print("🚀 Starting SMB Research Agent API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🌐 Web Interface: http://localhost:8000/static/index.html")
    print("🔍 API Endpoints: http://localhost:8000/")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )