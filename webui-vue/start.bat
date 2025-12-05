@echo off
@chcp 65001 >nul
:: Z-Image Trainer WebUI - One Click Start

echo.
echo ========================================
echo   Z-Image Trainer WebUI
echo ========================================
echo.

cd /d "%~dp0"

:: 设置嵌入版Python和Node.js路径（相对于项目根目录）
set "ROOT_DIR=%~dp0.."
set "PYTHON_EXE=%ROOT_DIR%\python\python.exe"
set "NPM_CMD=%ROOT_DIR%\nodejs\npm.cmd"

:: Check if dist exists
if not exist "dist" (
    echo [!] Frontend not built, building now...
    if not exist "node_modules" (
        echo [*] Installing npm dependencies...
        call "%NPM_CMD%" install
    )
    echo [*] Building frontend...
    call "%NPM_CMD%" run build
)

:: Start server
echo [*] Starting server on http://localhost:7860
"%PYTHON_EXE%" api/server.py --port 7860

pause
