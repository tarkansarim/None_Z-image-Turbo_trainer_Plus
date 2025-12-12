#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Server launcher script with proper path setup for embedded Python.
"""

import sys
import os

# Add API directory to path for 'core' and 'routers' imports
api_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, api_dir)

# Add src directory for zimage_trainer imports
project_root = os.path.dirname(os.path.dirname(api_dir))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

# Now import and run main
if __name__ == "__main__":
    from main import app
    import uvicorn
    import argparse
    from dotenv import load_dotenv
    from pathlib import Path
    
    # Load .env
    load_dotenv(Path(project_root) / ".env")
    
    # Parse arguments
    default_port = int(os.getenv("TRAINER_PORT", "9198"))
    default_host = os.getenv("TRAINER_HOST", "0.0.0.0")
    
    parser = argparse.ArgumentParser(description="Start None Trainer API Server")
    parser.add_argument("--host", type=str, default=default_host, help=f"Host to bind to (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port, help=f"Port to bind to (default: {default_port})")
    parser.add_argument("--dev", action="store_true", help="Development mode")
    args = parser.parse_args()
    
    print(f"Starting None Trainer on http://{args.host}:{args.port}")
    if args.dev:
        uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run(app, host=args.host, port=args.port)

