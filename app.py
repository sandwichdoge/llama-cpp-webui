"""FastAPI management API for llama-cpp-webui."""

import asyncio
import json
import traceback
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import builder, downloader, gguf_reader, server
from config import get_settings_path

app = FastAPI(title="llama-cpp-webui", version="1.0.0")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all errors return JSON, never HTML tracebacks."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Settings persistence ────────────────────────────────

def _load_all_settings() -> dict:
    p = get_settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_all_settings(data: dict):
    get_settings_path().write_text(json.dumps(data, indent=2))


def _save_model_settings(filename: str, settings: dict):
    all_s = _load_all_settings()
    all_s[filename] = settings
    _save_all_settings(all_s)


def _get_model_settings(filename: str) -> dict | None:
    return _load_all_settings().get(filename)


# ── Request schemas ──────────────────────────────────────

class BuildRequest(BaseModel):
    gcc_flags: list[str] = []

class DownloadRequest(BaseModel):
    url: str

class LoadRequest(BaseModel):
    model_path: str
    port: int = 5000
    n_gpu_layers: int = -1
    ctx_size: int = 4096
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


# ── UI ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ── Build endpoints ──────────────────────────────────────

@app.get("/api/build/status")
async def build_status():
    return builder.get_status()

@app.post("/api/build/start")
async def build_start(req: BuildRequest = BuildRequest()):
    status = builder.get_status()
    if status["status"] == "building":
        raise HTTPException(409, "Build already in progress")
    asyncio.create_task(builder.build(gcc_flags=req.gcc_flags))
    return {"message": "Build started"}


# ── Model endpoints ──────────────────────────────────────

@app.get("/api/models")
async def list_models():
    all_settings = _load_all_settings()
    models = downloader.list_models()
    for m in models:
        m["settings"] = all_settings.get(m["filename"])
        if not m["is_mmproj"]:
            m["meta"] = gguf_reader.read_metadata(m["path"])
    return {
        "models": models,
        "models_dir": str(downloader.get_models_dir()),
        "downloads": downloader.get_downloads(),
    }

@app.post("/api/models/download")
async def download_model(req: DownloadRequest):
    try:
        _, filename = downloader.parse_hf_url(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))

    asyncio.create_task(downloader.download_model(req.url))
    return {"message": f"Download started: {filename}", "filename": filename}

@app.delete("/api/models/{filename}")
async def delete_model(filename: str):
    if not downloader.delete_model(filename):
        raise HTTPException(404, "Model not found")
    return {"message": f"Deleted {filename}"}


# ── Server endpoints ─────────────────────────────────────

@app.get("/api/server/status")
async def server_status():
    return server.get_status()

@app.post("/api/server/start")
async def server_start(req: LoadRequest):
    try:
        await server.start(
            model_path=req.model_path,
            port=req.port,
            n_gpu_layers=req.n_gpu_layers,
            ctx_size=req.ctx_size,
            n_parallel=req.n_parallel,
            mmproj=req.mmproj,
            flash_attn=req.flash_attn,
            batch_size=req.batch_size,
            ubatch_size=req.ubatch_size,
            cpu_moe=req.cpu_moe,
            n_cpu_moe=req.n_cpu_moe,
            cache_type_k=req.cache_type_k,
            cache_type_v=req.cache_type_v,
            tensor_split=req.tensor_split,
            override_tensor=req.override_tensor,
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

    # Persist settings for this model
    filename = Path(req.model_path).name
    _save_model_settings(filename, {
        "port": req.port,
        "n_gpu_layers": req.n_gpu_layers,
        "ctx_size": req.ctx_size,
        "n_parallel": req.n_parallel,
        "mmproj": req.mmproj,
        "flash_attn": req.flash_attn,
        "batch_size": req.batch_size,
        "ubatch_size": req.ubatch_size,
        "cpu_moe": req.cpu_moe,
        "n_cpu_moe": req.n_cpu_moe,
        "cache_type_k": req.cache_type_k,
        "cache_type_v": req.cache_type_v,
        "tensor_split": req.tensor_split,
        "override_tensor": req.override_tensor,
    })

    return {"message": "Server starting…"}

@app.post("/api/server/stop")
async def server_stop():
    await server.stop()
    return {"message": "Server stopped"}