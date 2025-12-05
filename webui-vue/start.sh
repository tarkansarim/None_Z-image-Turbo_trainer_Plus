#!/bin/bash
# Z-Image Trainer WebUI - One Click Start

echo ""
echo "========================================"
echo "  Z-Image Trainer WebUI"
echo "========================================"
echo ""

cd "$(dirname "$0")"

# Check if dist exists
if [ ! -d "dist" ]; then
    echo "[!] Frontend not built, building now..."
    if [ ! -d "node_modules" ]; then
        echo "[*] Installing npm dependencies..."
        npm install
    fi
    echo "[*] Building frontend..."
    npm run build
fi

# Start server
echo "[*] Starting server on http://localhost:7860"
python api/server.py --port 7860
