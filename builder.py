"""Clone, pull, and build llama.cpp from source."""
from __future__ import annotations

import asyncio
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

from config import get_llama_cpp_dir, get_build_log_path, get_server_binary, get_data_dir

REPO_URL = "https://github.com/ggerganov/llama.cpp.git"

# Module-level build state — shared across requests
_build_state = {
    "status": "idle",        # idle | building | success | failed
    "progress": "",          # last line of build output
    "log": "",               # full log (tail)
    "started_at": None,
    "finished_at": None,
    "error": None,
    "commit": None,
    "commit_date": None,
}

_build_lock = asyncio.Lock()


def _get_git_info(repo_dir: Path) -> dict:
    """Get current commit hash and date."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir, text=True
        ).strip()
        date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci"], cwd=repo_dir, text=True
        ).strip()
        return {"commit": commit, "commit_date": date}
    except Exception:
        return {"commit": None, "commit_date": None}

def _get_driver_cuda_version() -> tuple[int, int] | None:
    """Return the max CUDA version the installed NVIDIA driver supports."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True,
        ).strip()
        # Parse the CUDA version from the nvidia-smi table header instead
        table = subprocess.check_output(
            ["nvidia-smi"], stderr=subprocess.STDOUT, text=True,
        )
        for line in table.splitlines():
            if "CUDA Version:" in line:
                # e.g. "| CUDA Version: 12.6     |"
                part = line.split("CUDA Version:")[1].strip().rstrip("|").strip()
                major, minor = part.split(".")
                return int(major), int(minor)
    except Exception:
        pass
    return None


def _get_nvcc_version() -> tuple[int, int] | None:
    """Return the installed CUDA toolkit version from nvcc."""
    try:
        out = subprocess.check_output(
            ["nvcc", "--version"], stderr=subprocess.STDOUT, text=True,
        )
        for line in out.splitlines():
            if "release" in line.lower():
                # e.g. "Cuda compilation tools, release 13.0, V13.0.88"
                part = line.split("release")[1].split(",")[0].strip()
                major, minor = part.split(".")
                return int(major), int(minor)
    except Exception:
        pass
    return None

def _detect_cmake_flags() -> list[str]:
    """Detect available hardware acceleration and return cmake flags."""
    flags = ["-DLLAMA_CURL=OFF"]  # we handle downloads ourselves

    # On Windows, force a single-config generator so the binary lands in
    # build/bin/ (flat) rather than build/bin/Release/ (MSVC default).
    # Prefer Ninja; fall back to NMake Makefiles.
    if sys.platform == "win32":
        try:
            subprocess.check_output(["ninja", "--version"], stderr=subprocess.STDOUT)
            flags += ["-G", "Ninja"]
        except (FileNotFoundError, subprocess.CalledProcessError):
            flags += ["-G", "NMake Makefiles"]

    # Check for CUDA
     # Check for CUDA
    try:
        subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)

        # Verify driver supports the installed toolkit version
        driver_cuda = _get_driver_cuda_version()
        nvcc_ver = _get_nvcc_version()
        if driver_cuda and nvcc_ver and nvcc_ver > driver_cuda:
            raise RuntimeError(
                f"CUDA toolkit {nvcc_ver[0]}.{nvcc_ver[1]} is installed, "
                f"but your NVIDIA driver only supports up to CUDA "
                f"{driver_cuda[0]}.{driver_cuda[1]}. "
                f"Please update your NVIDIA driver or install CUDA "
                f"{driver_cuda[0]}.{driver_cuda[1]} toolkit."
            )

        flags.append("-DGGML_CUDA=ON")

        # Detect GPU compute capability via nvidia-smi so we don't rely on
        # CMake's "native" detection, which can fail on Windows.
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                stderr=subprocess.STDOUT, text=True,
            )
            # e.g. "8.6\n" or "8.9\n7.5\n" for multi-GPU — keep all unique
            archs = sorted({line.strip().replace(".", "")
                            for line in out.strip().splitlines() if line.strip()})
            if archs:
                flags.append(f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(archs)}")
        except (FileNotFoundError, subprocess.CalledProcessError):
            # nvidia-smi not available; fall back to "all" so at least
            # common architectures are covered (slower build but works)
            flags.append("-DCMAKE_CUDA_ARCHITECTURES=all")

        return flags
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check for Vulkan SDK
    try:
        subprocess.check_output(["vulkaninfo", "--summary"], stderr=subprocess.STDOUT)
        flags.append("-DGGML_VULKAN=ON")
        return flags
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback — CPU only, but enable all available instruction sets via native march
    flags.append("-DLLAMA_NATIVE=ON")
    return flags


def get_status() -> dict:
    """Return current build status."""
    result = dict(_build_state)
    result["binary_exists"] = get_server_binary().is_file()
    result["repo_cloned"] = get_llama_cpp_dir().is_dir()

    # If we haven't populated commit info yet but repo exists, do it now
    if result["repo_cloned"] and not result["commit"]:
        info = _get_git_info(get_llama_cpp_dir())
        _build_state.update(info)
        result.update(info)

    return result


async def build(pull_only: bool = False, gcc_flags: list[str] | None = None):
    """Clone/pull and build llama.cpp. Runs in background."""
    if _build_lock.locked():
        return  # already building

    async with _build_lock:
        _build_state.update(status="building", progress="Starting…", log="",
                            started_at=datetime.datetime.now().isoformat(),
                            finished_at=None, error=None)

        try:
            await _run_build(gcc_flags=gcc_flags or [])
            info = _get_git_info(get_llama_cpp_dir())
            _build_state.update(status="success", progress="Build complete ✓",
                                finished_at=datetime.datetime.now().isoformat(), **info)
        except Exception as e:
            _build_state.update(status="failed", error=str(e),
                                finished_at=datetime.datetime.now().isoformat())


# Allowed GCC/Clang optimization flags (whitelist for safety).
# These are not applicable to MSVC on Windows.
_ALLOWED_GCC_FLAGS = {
    "-march=native",
    "-mtune=native",
    "-ffast-math",
    "-O3",
    "-flto",
}


def _build_gcc_flags_args(gcc_flags: list[str]) -> list[str]:
    """Validate and convert gcc_flags list into cmake -DCMAKE_*_FLAGS args.

    Returns an empty list on Windows because MSVC does not accept GCC flags.
    """
    if sys.platform == "win32":
        return []
    safe = [f for f in gcc_flags if f in _ALLOWED_GCC_FLAGS]
    if not safe:
        return []
    flags_str = " ".join(safe)
    return [
        f"-DCMAKE_C_FLAGS={flags_str}",
        f"-DCMAKE_CXX_FLAGS={flags_str}",
    ]


async def _run_build(gcc_flags: list[str] | None = None):
    repo_dir = get_llama_cpp_dir()

    # ── Clone or pull ───────────────────────────────────
    if not repo_dir.is_dir():
        _build_state["progress"] = "Cloning llama.cpp…"
        await _exec(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    else:
        _build_state["progress"] = "Pulling latest changes…"
        await _exec(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "master"])
        await _exec(["git", "-C", str(repo_dir), "reset", "--hard", "origin/master"])

    # ── Configure ────────────────────────────────────────
    build_dir = repo_dir / "build"
    build_dir.mkdir(exist_ok=True)

    cmake_flags = _detect_cmake_flags()
    extra_flags = _build_gcc_flags_args(gcc_flags or [])
    all_flags = cmake_flags + extra_flags
    _build_state["progress"] = f"Configuring… (flags: {' '.join(all_flags)})"

    cmake_cmd = ["cmake", "-B", str(build_dir), "-S", str(repo_dir),
                 "-DCMAKE_BUILD_TYPE=Release"] + all_flags
    await _exec(cmake_cmd)

    # ── Build ────────────────────────────────────────────
    import multiprocessing
    jobs = str(multiprocessing.cpu_count())
    _build_state["progress"] = f"Compiling with {jobs} jobs…"
    await _exec(["cmake", "--build", str(build_dir), "--config", "Release",
                 "-j", jobs, "--target", "llama-server"])

    # ── Verify ───────────────────────────────────────────
    if not get_server_binary().is_file():
        raise RuntimeError("Build succeeded but llama-server binary not found. "
                           "Check build log for details.")


async def _exec(cmd: list[str]):
    """Run a subprocess and stream output into build log."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    lines = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode(errors="replace").rstrip()
        lines.append(decoded)
        _build_state["progress"] = decoded
        # Keep last 200 lines in memory
        if len(lines) > 200:
            lines = lines[-200:]
        _build_state["log"] = "\n".join(lines)

    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                           + "\n".join(lines[-30:]))

    # Also persist full log to disk
    try:
        with open(get_build_log_path(), "a") as f:
            f.write(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}\n")
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass