@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: None Trainer - Windows 一键部署脚本
:: ============================================================================
:: 
:: 使用方法: 双击运行或在命令行执行 setup.bat
::
:: ============================================================================

echo ================================================
echo    None Trainer - 一键部署脚本
echo ================================================
echo.

:: 获取脚本目录
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: 设置嵌入版Python和Node.js路径
set "PYTHON_EXE=%SCRIPT_DIR%python\python.exe"
set "PIP_EXE=%SCRIPT_DIR%python\Scripts\pip.exe"
set "NODE_EXE=%SCRIPT_DIR%nodejs\node.exe"
set "NPM_CMD=%SCRIPT_DIR%nodejs\npm.cmd"

:: 检查嵌入版 Python
echo [1/8] 检查嵌入版 Python...
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到嵌入版 Python: %PYTHON_EXE%
    echo 请确保 python 目录存在
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('"%PYTHON_EXE%" --version 2^>^&1') do set PYTHON_VERSION=%%i
echo   嵌入版 Python: %PYTHON_VERSION%

:: 检查 CUDA
echo.
echo [2/8] 检查 CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo   [警告] 未检测到 NVIDIA GPU
) else (
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do set GPU_NAME=%%i
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=memory.total --format^=csv^,noheader 2^>nul') do set GPU_MEM=%%i
    echo   GPU: !GPU_NAME!
    echo   显存: !GPU_MEM!
)

:: 跳过虚拟环境（使用嵌入版Python）
echo.
echo [3/8] 使用嵌入版 Python（跳过虚拟环境）...
echo   嵌入版 Python 路径: %PYTHON_EXE%

:: 升级 pip
echo.
echo [4/8] 升级 pip...
"%PYTHON_EXE%" -m pip install --upgrade pip -q

:: 检查 PyTorch
echo.
echo [5/8] 检查 PyTorch...
"%PYTHON_EXE%" -c "import torch; print('PyTorch:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo   [错误] PyTorch 未安装！
    echo   请先手动安装 PyTorch，参考 README.md
    echo.
    echo   安装命令示例（使用嵌入版pip）：
    echo     CUDA 12.8: "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    echo     CUDA 12.1: "%PIP_EXE%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('"%PYTHON_EXE%" -c "import torch; print(torch.__version__)"') do echo   PyTorch 已安装: %%i
for /f "tokens=*" %%i in ('"%PYTHON_EXE%" -c "import torch; print(torch.version.cuda if torch.cuda.is_available() else 'CPU')"') do echo   CUDA 版本: %%i

:: 检查 Flash Attention
echo.
"%PYTHON_EXE%" -c "import flash_attn; print(flash_attn.__version__)" 2>nul
if errorlevel 1 (
    echo   Flash Attention 未安装（可选，建议安装以提升性能）
) else (
    for /f "tokens=*" %%i in ('"%PYTHON_EXE%" -c "import flash_attn; print(flash_attn.__version__)"') do echo   Flash Attention 已安装: %%i
)

:: 安装依赖
echo.
echo [6/8] 安装 Python 依赖...
"%PIP_EXE%" install -r requirements.txt -q

:: 安装 diffusers 最新版
echo   安装 diffusers (git 最新版)...
"%PIP_EXE%" install git+https://github.com/huggingface/diffusers.git -q

:: 安装本项目
"%PIP_EXE%" install -e . -q

:: 创建 .env 文件
if not exist ".env" (
    copy env.example .env >nul
    echo   已创建 .env 配置文件
)

:: 检查嵌入版 Node.js
echo.
echo [7/8] 检查嵌入版 Node.js...
if not exist "%NODE_EXE%" (
    echo   [警告] 未找到嵌入版 Node.js: %NODE_EXE%
    echo   请确保 nodejs 目录存在
    echo.
    echo   跳过前端构建，后端服务仍可运行
    echo.
    echo ================================================
    echo 后端部署完成！
    echo ================================================
    echo.
    echo 后续步骤:
    echo   1. 确保 nodejs 目录存在后运行: cd webui-vue ^&^& npm install ^&^& npm run build
    echo   2. 编辑 .env 配置模型路径
    echo   3. 运行 start.bat 启动服务
    pause
    exit /b 0
)
for /f "tokens=*" %%i in ('"%NODE_EXE%" --version') do echo   嵌入版 Node.js: %%i

:: 构建前端
echo.
echo [8/8] 构建前端...
cd /d "%SCRIPT_DIR%webui-vue"

if not exist "node_modules" (
    echo   安装前端依赖...
    call "%NPM_CMD%" install --silent
)

echo   构建前端...
call "%NPM_CMD%" run build --silent

cd /d "%SCRIPT_DIR%"
echo   前端构建完成

:: 完成
echo.
echo ================================================
echo ✅ 部署完成！
echo ================================================
echo.
echo 后续步骤:
echo   1. 编辑 .env 配置模型路径
echo   2. 运行 start.bat 启动服务
echo.
echo 常用命令:
echo   start.bat          启动 Web UI
echo   start.bat --help   查看帮助
echo.
pause

