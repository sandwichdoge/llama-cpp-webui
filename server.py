"""Manage the llama-server inference process."""
from __future__ import annotations

import asyncio
import datetime
import logging
import shlex
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from config import get_server_binary, save_model_settings

log = logging.getLogger(__name__)


class LoadRequest(BaseModel):
    """API + internal launch spec for llama-server."""
    model_path: str
    port: int = 5000
    n_gpu_layers: int = -1
    ctx_size: int = 32768
    n_parallel: int = 1
    # Multimodal
    mmproj: str = ""
    # Advanced / MoE
    flash_attn: str = "auto"
    batch_size: int = 2048
    ubatch_size: int = 512
    cpu_moe: bool = False
    n_cpu_moe: int = 0
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    tensor_split: str = ""
    override_tensor: str = ""
    # Extra command-line arguments
    extra_args: str = ""
    # IDs of presets currently toggled active in the UI. Persisted with model
    # settings so reloads restore the highlighted state. The preset args
    # themselves are already folded into `extra_args` by the frontend.
    active_preset_ids: list[str] = Field(default_factory=list)
    # Speculative decoding draft model (MTP)
    draft_model_path: str = ""
    draft_gpu_layers: int = -1
    draft_max: int = 3
    draft_p_min: float = 0.0
    # Inference bind host. Defaults to loopback for safety — set to "0.0.0.0"
    # explicitly if you want the inference port reachable from the LAN.
    bind_host: str = "127.0.0.1"


# Server process state
_state: dict = {
    "status": "stopped",      # stopped | starting | running | error
    "model": None,             # loaded model filename
    "model_path": None,        # full path to model
    "port": 5000,              # inference port
    "pid": None,
    "error": None,
    "params": {},              # launch params
    "started_at": None,
    "cmd": "",                 # the exact command launched
    "log": "",                 # last N lines of server output
    "model_settings": {},      # full LoadRequest persisted at successful launch
}

_process: asyncio.subprocess.Process | None = None
_monitor_task: asyncio.Task | None = None
_log_task: asyncio.Task | None = None
_log_lines: list[str] = []
_LOG_MAX = 150
_health_client: httpx.AsyncClient | None = None


def _persist_model_settings() -> None:
    """Save per-model settings to disk after successful model load."""
    filename = _state.get("model")
    settings = _state.get("model_settings")
    if filename and settings:
        save_model_settings(filename, settings)


def get_status() -> dict:
    result = dict(_state)
    result["log"] = "\n".join(_log_lines[-_LOG_MAX:])
    return result


def _build_command(req: LoadRequest, binary: Path, model: Path) -> list[str]:
    """Assemble the llama-server argv from a launch request."""
    cmd = [
        str(binary),
        "-m", str(model),
        "--port", str(req.port),
        "--host", req.bind_host,
        "-ngl", str(req.n_gpu_layers),
        "-c", str(req.ctx_size),
        "-np", str(req.n_parallel),
        "-b", str(req.batch_size),
        "-ub", str(req.ubatch_size),
        "-fa", req.flash_attn,
        "-ctk", req.cache_type_k,
        "-ctv", req.cache_type_v,
        "--jinja",
        "--metrics",
        "--fit-target", "300",
    ]

    if req.cpu_moe:
        cmd += ["-cmoe"]
    elif req.n_cpu_moe > 0:
        cmd += ["-ncmoe", str(req.n_cpu_moe)]

    if req.mmproj.strip():
        mmproj_path = Path(req.mmproj.strip())
        if mmproj_path.is_file():
            cmd += ["--mmproj", str(mmproj_path)]

    if req.draft_model_path.strip():
        draft_path = Path(req.draft_model_path.strip())
        if draft_path.is_file():
            cmd += ["--spec-type", "mtp",
                    "-md", str(draft_path),
                    "-ngld", str(req.draft_gpu_layers),
                    "--draft-max", str(req.draft_max),
                    "--draft-p-min", str(req.draft_p_min)]

    if req.tensor_split.strip():
        cmd += ["-ts", req.tensor_split.strip()]

    if req.override_tensor.strip():
        cmd += ["-ot", req.override_tensor.strip()]

    # Extra command-line arguments. The frontend already folds any active
    # preset args into this string when the user toggles a preset on; we only
    # persist `active_preset_ids` for highlight restore, not for re-expansion.
    if req.extra_args.strip():
        try:
            cmd.extend(shlex.split(req.extra_args.strip()))
        except ValueError as e:
            raise ValueError(f"Invalid extra_args: {e}")

    return cmd


async def start(req: LoadRequest) -> None:
    """Start llama-server with the given model."""
    global _process, _monitor_task, _log_task

    if _state["status"] in ("running", "starting"):
        raise RuntimeError("Server already running. Stop it first.")

    binary = get_server_binary()
    if not binary.is_file():
        raise FileNotFoundError("llama-server binary not found. Build llama.cpp first.")

    model = Path(req.model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {req.model_path}")

    cmd = _build_command(req, binary, model)

    _state.update(
        status="starting",
        model=model.name,
        model_path=str(model),
        port=req.port,
        error=None,
        params={
            "n_gpu_layers": req.n_gpu_layers,
            "ctx_size": req.ctx_size,
            "n_parallel": req.n_parallel,
        },
        model_settings=req.model_dump(),
    )

    try:
        _process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(binary.parent),
        )
        _state["pid"] = _process.pid
        _state["cmd"] = " ".join(cmd)
        _log_lines.clear()

        _log_task = asyncio.create_task(_log_reader())
        _monitor_task = asyncio.create_task(_health_monitor(req.port))

    except Exception as e:
        log.exception("Failed to spawn llama-server")
        _state.update(status="error", error=str(e), pid=None, started_at=None)
        raise


async def stop() -> None:
    """Stop the running llama-server process."""
    global _process, _monitor_task, _log_task, _health_client

    if _monitor_task:
        _monitor_task.cancel()
        _monitor_task = None

    if _log_task:
        _log_task.cancel()
        _log_task = None

    if _process and _process.returncode is None:
        _process.terminate()
        try:
            await asyncio.wait_for(_process.wait(), timeout=10)
        except asyncio.TimeoutError:
            _process.kill()
            await _process.wait()

    if _health_client:
        await _health_client.aclose()
        _health_client = None

    _process = None
    _state.update(status="stopped", model=None, model_path=None, pid=None,
                  error=None, started_at=None, cmd="", model_settings={})


async def shutdown() -> None:
    """Gracefully shut down server resources on app exit."""
    await stop()


async def _log_reader() -> None:
    """Read llama-server stdout and store in _log_lines."""
    try:
        while _process and _process.returncode is None and _process.stdout is not None:
            line = await _process.stdout.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").rstrip()
            _log_lines.append(decoded)
            if len(_log_lines) > _LOG_MAX * 2:
                del _log_lines[:_LOG_MAX]
    except asyncio.CancelledError:
        pass


async def _health_monitor(port: int) -> None:
    """Poll the llama-server /health endpoint until it's ready, then watch it."""
    global _health_client
    url = f"http://127.0.0.1:{port}/health"

    _health_client = httpx.AsyncClient()

    # Wait for server to become healthy (model loading can take a while)
    for _ in range(600):  # up to 5 minutes at 0.5s cadence
        if _process is None or _process.returncode is not None:
            exit_code = _process.returncode if _process else "unknown"
            _state.update(status="error", error=f"Process exited (code {exit_code})")
            return

        try:
            r = await _health_client.get(url, timeout=2)
            data = r.json()
            if data.get("status") == "ok":
                _state.update(status="running",
                              started_at=datetime.datetime.now().isoformat())
                _persist_model_settings()
                break
            elif data.get("status") == "loading model":
                _state["status"] = "starting"
        except Exception:
            pass

        await asyncio.sleep(0.5)
    else:
        _state.update(status="error", error="Server failed to become healthy within 5 minutes")
        return

    # Ongoing health watch
    while True:
        await asyncio.sleep(5)
        if _process is None or _process.returncode is not None:
            _state.update(status="error", error="Process exited unexpectedly")
            return

        try:
            r = await _health_client.get(url, timeout=5)
            if r.status_code != 200:
                _state.update(status="error", error=f"Health check returned {r.status_code}")
        except Exception as e:
            _state.update(status="error", error=f"Health check failed: {e}")
