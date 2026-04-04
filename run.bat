@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

:: ══════════════════════════════════════════════════════════
::  Version pins  (bump these when updating)
:: ══════════════════════════════════════════════════════════
set PY_VER=3.12.8
set GIT_VER=2.47.1
set CMAKE_VER=3.31.4
set NINJA_VER=1.12.1
set W64DEV_VER=2.0.0

set "TOOLS=%~dp0.tools"
set INSTALLED_ANY=0

:: ── Try to activate an existing MSVC install ─────────────
call :activate_msvc

:: ── Check each dependency; install if missing ────────────
where python  >nul 2>&1 || call :install_python
where git     >nul 2>&1 || call :install_git
where cmake   >nul 2>&1 || call :install_cmake
where ninja   >nul 2>&1 || call :install_ninja
(where cl >nul 2>&1 || where g++ >nul 2>&1 || where clang++ >nul 2>&1) || call :install_compiler

:: ── Pick up any PATH changes the installers wrote ────────
if !INSTALLED_ANY!==1 call :refresh_path

:: Add portable .tools dirs to the *session* PATH
if exist "%TOOLS%\cmake\bin"     set "PATH=%TOOLS%\cmake\bin;!PATH!"
if exist "%TOOLS%\ninja"         set "PATH=%TOOLS%\ninja;!PATH!"
if exist "%TOOLS%\w64devkit\bin" set "PATH=%TOOLS%\w64devkit\bin;!PATH!"

:: Re-check MSVC — Build Tools may have just been installed
call :activate_msvc

:: ── Final verification ───────────────────────────────────
set "FAIL="
where python >nul 2>&1 || set "FAIL=!FAIL! python"
where git    >nul 2>&1 || set "FAIL=!FAIL! git"
where cmake  >nul 2>&1 || set "FAIL=!FAIL! cmake"
(where cl >nul 2>&1 || where g++ >nul 2>&1 || where clang++ >nul 2>&1) || set "FAIL=!FAIL! C++-compiler"

if not "!FAIL!"=="" (
    echo.
    echo   [ERROR] Auto-install failed for:!FAIL!
    echo       Please install manually and re-run:
    echo         Python   : https://www.python.org/downloads/
    echo         Git      : https://git-scm.com/download/win
    echo         CMake    : https://cmake.org/download/
    echo         Compiler : https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo.
    exit /b 1
)

:: ── GPU status (first run only) ──────────────────────────
if not exist .venv (
    where nvcc       >nul 2>&1 && echo   CUDA detected
    where vulkaninfo >nul 2>&1 && echo   Vulkan detected
    if not defined CUDA_PATH if not defined VULKAN_SDK (
        echo   No GPU toolkit found ^(CUDA / Vulkan^) -- will build CPU-only
    )
)

:: ── Virtual environment & run ────────────────────────────
if not exist .venv (
    echo   Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
pip install -q -r requirements.txt

python run.py %*
exit /b 0


:: ══════════════════════════════════════════════════════════
::                    S U B R O U T I N E S
:: ══════════════════════════════════════════════════════════

:: ── Re-read PATH from the registry ──────────────────────
::    Installers update the registry, but the current cmd
::    session still has the old PATH.  This fixes that.
:refresh_path
    :: Append newly-installed paths to the session PATH.
    :: (Duplicates are harmless; nuking the existing PATH is not.)
    for /f "tokens=2*" %%A in (
        'reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul'
    ) do set "PATH=!PATH!;%%B"
    for /f "tokens=2*" %%A in (
        'reg query "HKCU\Environment" /v Path 2^>nul'
    ) do set "PATH=!PATH!;%%B"
    goto :eof

:: ── Activate MSVC toolchain if installed but not on PATH ─
:activate_msvc
    where cl >nul 2>&1 && goto :eof
    set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if not exist "!VSWHERE!" goto :eof
    :: First try: only installations that have VC tools
    set "VSDIR="
    for /f "delims=" %%i in (
        '"!VSWHERE!" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul'
    ) do set "VSDIR=%%i"
    :: Fallback: any VS/Build Tools installation (might have VC in a non-standard layout)
    if not defined VSDIR (
        for /f "delims=" %%i in (
            '"!VSWHERE!" -latest -products * -property installationPath 2^>nul'
        ) do set "VSDIR=%%i"
    )
    if not defined VSDIR goto :eof
    if exist "!VSDIR!\VC\Auxiliary\Build\vcvarsall.bat" (
        echo   Activating MSVC toolchain...
        call "!VSDIR!\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
    )
    goto :eof


:: ──────────────────────────────────────────────────────────
::  Python   (per-user install — no admin required)
:: ──────────────────────────────────────────────────────────
:install_python
    echo.
    echo   [1/5] Python not found -- installing %PY_VER%...
    where winget >nul 2>&1 || goto :install_python_curl
    winget install -e --id Python.Python.3.12 -v %PY_VER% ^
        --accept-source-agreements --accept-package-agreements -h
    set INSTALLED_ANY=1
    goto :eof
:install_python_curl
    echo         Downloading installer...
    curl.exe -fSL --retry 3 -o "%TEMP%\python-setup.exe" ^
        "https://www.python.org/ftp/python/%PY_VER%/python-%PY_VER%-amd64.exe"
    if !errorlevel! neq 0 (echo         Download failed. & goto :eof)
    echo         Running installer ^(per-user, no admin needed^)...
    start /wait "" "%TEMP%\python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1
    del "%TEMP%\python-setup.exe" 2>nul
    set INSTALLED_ANY=1
    goto :eof


:: ──────────────────────────────────────────────────────────
::  Git   (system install — may trigger UAC prompt)
:: ──────────────────────────────────────────────────────────
:install_git
    echo.
    echo   [2/5] Git not found -- installing %GIT_VER%...
    where winget >nul 2>&1 || goto :install_git_curl
    winget install -e --id Git.Git -v %GIT_VER% ^
        --accept-source-agreements --accept-package-agreements -h
    set INSTALLED_ANY=1
    goto :eof
:install_git_curl
    echo         Downloading installer...
    curl.exe -fSL --retry 3 -o "%TEMP%\git-setup.exe" ^
        "https://github.com/git-for-windows/git/releases/download/v%GIT_VER%.windows.1/Git-%GIT_VER%-64-bit.exe"
    if !errorlevel! neq 0 (echo         Download failed. & goto :eof)
    echo         Running installer ^(silent, may request admin^)...
    start /wait "" "%TEMP%\git-setup.exe" /VERYSILENT /NORESTART /SP-
    del "%TEMP%\git-setup.exe" 2>nul
    set INSTALLED_ANY=1
    goto :eof


:: ──────────────────────────────────────────────────────────
::  CMake   (portable zip — no admin, no installer)
:: ──────────────────────────────────────────────────────────
:install_cmake
    echo.
    echo   [3/5] CMake not found -- installing %CMAKE_VER% ^(portable^)...
    if not exist "%TOOLS%\cmake" mkdir "%TOOLS%\cmake"
    curl.exe -fSL --retry 3 -o "%TEMP%\cmake.zip" ^
        "https://github.com/Kitware/CMake/releases/download/v%CMAKE_VER%/cmake-%CMAKE_VER%-windows-x86_64.zip"
    if !errorlevel! neq 0 (echo         Download failed. & goto :eof)
    echo         Extracting...
    tar -xf "%TEMP%\cmake.zip" -C "%TOOLS%\cmake" --strip-components=1 2>nul
    del "%TEMP%\cmake.zip" 2>nul
    goto :eof


:: ──────────────────────────────────────────────────────────
::  Ninja   (portable zip — single exe, no admin)
:: ──────────────────────────────────────────────────────────
:install_ninja
    echo.
    echo   [4/5] Ninja not found -- installing %NINJA_VER% ^(portable^)...
    if not exist "%TOOLS%\ninja" mkdir "%TOOLS%\ninja"
    curl.exe -fSL --retry 3 -o "%TEMP%\ninja.zip" ^
        "https://github.com/ninja-build/ninja/releases/download/v%NINJA_VER%/ninja-win.zip"
    if !errorlevel! neq 0 (echo         Download failed. & goto :eof)
    tar -xf "%TEMP%\ninja.zip" -C "%TOOLS%\ninja" 2>nul
    del "%TEMP%\ninja.zip" 2>nul
    goto :eof


:: ──────────────────────────────────────────────────────────
::  C++ Compiler
::    Stage A:  Try VS Build Tools (--force to fix stale installs)
::    Stage B:  Portable MinGW via w64devkit (no admin needed)
:: ──────────────────────────────────────────────────────────
:install_compiler
    echo.
    echo   [5/5] No C++ compiler found.
    echo.

    :: ── Stage A: VS Build Tools ──────────────────────────
    echo         --- Stage A: Visual Studio Build Tools ---
    net session >nul 2>&1
    if !errorlevel! neq 0 (
        echo         [info] Not running as admin. UAC prompt will appear.
    )

    where winget >nul 2>&1 || goto :install_compiler_vs_curl

    echo         Using winget...
    winget install -e --id Microsoft.VisualStudio.2022.BuildTools ^
        --override "--quiet --wait --force --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" ^
        --accept-source-agreements --accept-package-agreements
    set "VS_EXIT=!errorlevel!"
    echo         winget exit code: !VS_EXIT!
    if !VS_EXIT! equ 0 (set INSTALLED_ANY=1& goto :eof)
    if !VS_EXIT! equ 3010 (echo         Success, reboot needed later. & set INSTALLED_ANY=1& goto :eof)
    echo         winget failed. Trying direct download...

:install_compiler_vs_curl
    curl.exe -fSL --retry 3 -o "%TEMP%\vs_BuildTools.exe" ^
        "https://aka.ms/vs/17/release/vs_BuildTools.exe"
    if !errorlevel! neq 0 (
        echo         Download failed. Skipping to Stage B...
        goto :install_compiler_mingw
    )

    echo         Launching VS installer with --force ^(UAC prompt will appear^)...
    echo         This may take 5-15 minutes.
    echo.

    powershell -Command ^
        "try { $p = Start-Process -FilePath '%TEMP%\vs_BuildTools.exe' -ArgumentList '--quiet --wait --force --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended' -Verb RunAs -Wait -PassThru; exit $p.ExitCode } catch { Write-Host '         UAC cancelled or elevation failed:' $_.Exception.Message; exit 1 }"
    set "VS_EXIT=!errorlevel!"
    del "%TEMP%\vs_BuildTools.exe" 2>nul

    echo.
    if !VS_EXIT! equ 0 (
        echo         VS Build Tools installed successfully.
        set INSTALLED_ANY=1
        goto :eof
    )
    if !VS_EXIT! equ 3010 (
        echo         VS Build Tools installed. Reboot needed later.
        set INSTALLED_ANY=1
        goto :eof
    )

    echo         VS installer exited with code !VS_EXIT!.
    echo         Logs: %TEMP%\dd_bootstrapper_*.log
    echo               %TEMP%\dd_setup_*.log
    echo.

    :: ── Stage B: Portable MinGW (w64devkit) ──────────────
:install_compiler_mingw
    echo         --- Stage B: Portable MinGW ^(w64devkit %W64DEV_VER%^) ---
    echo         No admin needed. ~70 MB download.
    echo.

    if not exist "%TOOLS%\w64devkit" mkdir "%TOOLS%\w64devkit"
    curl.exe -fSL --retry 3 -o "%TEMP%\w64devkit.zip" ^
        "https://github.com/skeeto/w64devkit/releases/download/v%W64DEV_VER%/w64devkit-%W64DEV_VER%-x64.zip"
    if !errorlevel! neq 0 (
        echo         Download failed.
        echo         Try: https://github.com/skeeto/w64devkit/releases
        goto :eof
    )

    echo         Extracting to .tools\w64devkit...
    tar -xf "%TEMP%\w64devkit.zip" -C "%TOOLS%" 2>nul
    del "%TEMP%\w64devkit.zip" 2>nul

    if exist "%TOOLS%\w64devkit\bin\g++.exe" (
        echo         w64devkit installed successfully ^(g++ available^).
    ) else (
        echo         Extraction may have failed. Check .tools\w64devkit\
    )
    goto :eof
