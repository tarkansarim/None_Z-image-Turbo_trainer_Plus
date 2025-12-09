@echo off
echo ============================================
echo   Z-Image Trainer - Process Cleanup
echo ============================================
echo.

:: Find and kill Python processes from this project directory
for /f "tokens=2" %%i in ('wmic process where "commandline like '%%None_Z-image%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo Killing process %%i...
    taskkill /F /PID %%i 2>nul
)

:: Also check for any python processes using the embedded Python
for /f "tokens=2" %%i in ('wmic process where "executablepath like '%%None_Z-image-Turbo_trainer%%python%%'" get processid 2^>nul ^| findstr /r "[0-9]"') do (
    echo Killing embedded Python process %%i...
    taskkill /F /PID %%i 2>nul
)

echo.
echo Cleanup complete! Checking GPU memory...
echo.
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>nul || echo nvidia-smi not available
echo.
pause
