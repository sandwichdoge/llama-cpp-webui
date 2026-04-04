@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:: ── Check system dependencies ────────────────────────────
set MISSING=

where python >nul 2>&1 || set MISSING=!MISSING! python
where git    >nul 2>&1 || set MISSING=!MISSING! git
where cmake  >nul 2>&1 || set MISSING=!MISSING! cmake

:: Check for a C++ compiler (cl, g++, or clang++)
where cl     >nul 2>&1 || where g++   >nul 2>&1 || where clang++ >nul 2>&1 || (
    set MISSING=!MISSING! "C++ compiler (cl/g++/clang++)"
)

if not "!MISSING!"=="" (
    echo.
    echo   ^!  Missing system dependencies:!MISSING!
    echo.
    echo   Install the following and re-run:
    echo     Python   : https://www.python.org/downloads/
    echo     Git      : https://git-scm.com/download/win
    echo     CMake    : https://cmake.org/download/
    echo     Compiler : Visual Studio Build Tools  ^(recommended^)
    echo                https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo     Ninja    : winget install Ninja-build.Ninja  ^(optional but faster^)
    echo.
    echo   For GPU acceleration:
    echo     NVIDIA CUDA : https://developer.nvidia.com/cuda-downloads
    echo     Vulkan SDK  : https://vulkan.lunarg.com/sdk/home#windows
    echo.
    exit /b 1
)

:: ── Report GPU status on first run ───────────────────────
if not exist .venv (
    where nvcc       >nul 2>&1 && echo   CUDA detected
    where vulkaninfo >nul 2>&1 && echo   Vulkan detected
    if not defined CUDA_PATH if not defined VULKAN_SDK (
        echo   No GPU toolkit found ^(CUDA/Vulkan^) -- will build CPU-only
    )
)

:: ── Create venv if needed ────────────────────────────────
if not exist .venv (
    echo   Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

python run.py %*
