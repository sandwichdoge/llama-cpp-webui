"""Clone tabbyAPI and install it into its own venv (torch + exllamav3)."""
from __future__ import annotations

import asyncio
import datetime
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from builder import _get_driver_cuda_version
from config import get_tabby_dir, get_tabby_install_log_path

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/theroyallab/tabbyAPI.git"

# Module-level install state — shared across requests
_install_state = {
    "status": "idle",        # idle | installing | success | failed
    "progress": "",          # last line of install output
    "log": "",               # full log (tail)
    "started_at": None,
    "finished_at": None,
    "error": None,
}

_install_lock = asyncio.Lock()


def venv_python() -> Path:
    if sys.platform == "win32":
        return get_tabby_dir() / "venv" / "Scripts" / "python.exe"
    return get_tabby_dir() / "venv" / "bin" / "python"


def _install_marker() -> Path:
    return get_tabby_dir() / ".webui_installed"


def is_installed() -> bool:
    return _install_marker().is_file() and venv_python().is_file()


def get_status() -> dict:
    result = dict(_install_state)
    result["installed"] = is_installed()
    result["repo_cloned"] = get_tabby_dir().is_dir()
    return result


def is_installing() -> bool:
    return _install_lock.locked()


def _detect_gpu_extra() -> str:
    """Pick the pip extra matching the local GPU stack ('' = CPU/unknown)."""
    cuda = _get_driver_cuda_version()
    if cuda:
        return "cu13" if cuda[0] >= 13 else "cu12"
    if shutil.which("rocm-smi"):
        return "amd"
    return ""


async def install():
    """Clone/pull tabbyAPI and pip-install it into a venv. Runs in background.

    Safe under concurrent invocation: second caller silently returns without
    clobbering the in-flight install's state (same pattern as builder.build).
    """
    if _install_lock.locked():
        return

    async with _install_lock:
        _install_state.update(status="installing", progress="Starting…", log="",
                              started_at=datetime.datetime.now().isoformat(),
                              finished_at=None, error=None)
        try:
            await _run_install()
            _install_state.update(status="success", progress="Install complete ✓",
                                  finished_at=datetime.datetime.now().isoformat())
        except Exception as e:
            log.exception("tabbyAPI install failed")
            _install_state.update(status="failed", error=str(e),
                                  finished_at=datetime.datetime.now().isoformat())


async def _run_install():
    repo_dir = get_tabby_dir()

    # ── Clone or pull ───────────────────────────────────
    if not repo_dir.is_dir():
        _install_state["progress"] = "Cloning tabbyAPI…"
        await _exec(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)])
    else:
        _install_state["progress"] = "Pulling latest changes…"
        await _exec(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", "main"])
        await _exec(["git", "-C", str(repo_dir), "reset", "--hard", "origin/main"])

    # ── Create venv ──────────────────────────────────────
    if not venv_python().is_file():
        _install_state["progress"] = "Creating venv…"
        await _exec([sys.executable, "-m", "venv", str(repo_dir / "venv")])

    # ── Install dependencies ─────────────────────────────
    extra = _detect_gpu_extra()
    if not extra:
        log.warning("No NVIDIA/AMD GPU stack detected; installing tabbyAPI without a GPU extra")
    target = f".[{extra}]" if extra else "."
    _install_state["progress"] = f"Installing dependencies ({target})… this downloads several GB"
    await _exec([str(venv_python()), "-m", "pip", "install", "-U", target], cwd=repo_dir)

    _install_marker().touch()


async def _exec(cmd: list[str], cwd: Path | None = None):
    """Run a subprocess and stream output into the install log."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
    )
    assert proc.stdout is not None  # PIPE was requested above
    lines: list[str] = []
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        decoded = line.decode(errors="replace").rstrip()
        lines.append(decoded)
        _install_state["progress"] = decoded
        # Keep last 200 lines in memory
        if len(lines) > 200:
            lines = lines[-200:]
        _install_state["log"] = "\n".join(lines)

    await proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                           + "\n".join(lines[-30:]))

    # Also persist full log to disk
    try:
        with open(get_tabby_install_log_path(), "a") as f:
            f.write(f"\n{'='*60}\n$ {' '.join(cmd)}\n{'='*60}\n")
            f.write("\n".join(lines) + "\n")
    except OSError:
        log.warning("Could not append to install log at %s", get_tabby_install_log_path())
