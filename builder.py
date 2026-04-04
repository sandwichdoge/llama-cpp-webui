"""Clone, pull, and build llama.cpp from source."""

import asyncio
import datetime
import json
import subprocess
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


def _detect_cmake_flags() -> list[str]:
    """Detect available hardware acceleration and return cmake flags."""
    flags = ["-DLLAMA_CURL=OFF"]  # we handle downloads ourselves

    # Check for CUDA
    try:
        subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        flags.append("-DGGML_CUDA=ON")
        return flags  # Prefer CUDA over others
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Check for ROCm / HIP
    if Path("/opt/rocm").exists():
        flags.append("-DGGML_HIP=ON")
        return flags

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


async def build(pull_only: bool = False):
    """Clone/pull and build llama.cpp. Runs in background."""
    if _build_lock.locked():
        return  # already building

    async with _build_lock:
        _build_state.update(status="building", progress="Starting…", log="",
                            started_at=datetime.datetime.now().isoformat(),
                            finished_at=None, error=None)

        try:
            await _run_build()
            info = _get_git_info(get_llama_cpp_dir())
            _build_state.update(status="success", progress="Build complete ✓",
                                finished_at=datetime.datetime.now().isoformat(), **info)
        except Exception as e:
            _build_state.update(status="failed", error=str(e),
                                finished_at=datetime.datetime.now().isoformat())


async def _run_build():
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
    _build_state["progress"] = f"Configuring… (flags: {' '.join(cmake_flags)})"

    cmake_cmd = ["cmake", "-B", str(build_dir), "-S", str(repo_dir),
                 "-DCMAKE_BUILD_TYPE=Release"] + cmake_flags
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
