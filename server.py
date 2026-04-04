"""Manage the llama-server inference process."""

import asyncio
import signal
import time
from pathlib import Path

import httpx

from config import get_server_binary

# Server process state
_state = {
    "status": "stopped",      # stopped | starting | running | error
    "model": None,             # loaded model filename
    "model_path": None,        # full path to model
    "port": 8080,              # inference port
    "pid": None,
    "error": None,
    "params": {},              # launch params
    "started_at": None,
}

_process: asyncio.subprocess.Process | None = None
_monitor_task: asyncio.Task | None = None


def get_status() -> dict:
    return dict(_state)


async def start(model_path: str, port: int = 8080, n_gpu_layers: int = -1,
                ctx_size: int = 4096, n_parallel: int = 1):
    """Start llama-server with the given model."""
    global _process, _monitor_task

    if _state["status"] == "running":
        raise RuntimeError("Server already running. Stop it first.")

    binary = get_server_binary()
    if not binary.is_file():
        raise FileNotFoundError("llama-server binary not found. Build llama.cpp first.")

    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    _state.update(
        status="starting",
        model=model.name,
        model_path=str(model),
        port=port,
        error=None,
        params={
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "n_parallel": n_parallel,
        },
    )

    cmd = [
        str(binary),
        "-m", str(model),
        "--port", str(port),
        "--host", "0.0.0.0",
        "-ngl", str(n_gpu_layers),
        "-c", str(ctx_size),
        "-np", str(n_parallel),
        "--metrics",
    ]

    try:
        _process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        _state["pid"] = _process.pid

        # Start background health-check monitor
        _monitor_task = asyncio.create_task(_health_monitor(port))

    except Exception as e:
        _state.update(status="error", error=str(e))
        raise


async def stop():
    """Stop the running llama-server process."""
    global _process, _monitor_task

    if _monitor_task:
        _monitor_task.cancel()
        _monitor_task = None

    if _process and _process.returncode is None:
        _process.terminate()
        try:
            await asyncio.wait_for(_process.wait(), timeout=10)
        except asyncio.TimeoutError:
            _process.kill()
            await _process.wait()

    _process = None
    _state.update(status="stopped", model=None, model_path=None, pid=None,
                  error=None, started_at=None)


async def _health_monitor(port: int):
    """Poll the llama-server /health endpoint until it's ready, then watch it."""
    url = f"http://127.0.0.1:{port}/health"

    # Wait for server to become healthy (model loading can take a while)
    for attempt in range(600):  # up to 5 minutes
        if _process is None or _process.returncode is not None:
            exit_code = _process.returncode if _process else "unknown"
            _state.update(status="error", error=f"Process exited (code {exit_code})")
            return

        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=2)
                data = r.json()
                if data.get("status") == "ok":
                    _state.update(status="running",
                                  started_at=__import__("datetime").datetime.now().isoformat())
                    break
                elif data.get("status") == "loading model":
                    _state["status"] = "starting"
        except Exception:
            pass

        await asyncio.sleep(0.5)
    else:
        _state.update(status="error", error="Server failed to become healthy within 5 minutes")
        return

    # Now monitor ongoing health
    while True:
        await asyncio.sleep(5)
        if _process is None or _process.returncode is not None:
            _state.update(status="error", error="Process exited unexpectedly")
            return

        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, timeout=5)
                if r.status_code != 200:
                    _state.update(status="error", error=f"Health check returned {r.status_code}")
        except Exception as e:
            _state.update(status="error", error=f"Health check failed: {e}")
