"""FastAPI management API for llama-cpp-webui."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import builder, downloader, gguf_reader, presets, server
from config import load_all_settings

presets.seed_if_empty()

app = FastAPI(title="llama-cpp-webui", version="1.0.0")


@app.on_event("shutdown")
async def on_shutdown():
    """Gracefully shut down all background resources."""
    await server.shutdown()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure all errors return JSON, never HTML tracebacks."""
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ── Request schemas ──────────────────────────────────────

class BuildRequest(BaseModel):
    gcc_flags: list[str] = []

class DownloadRequest(BaseModel):
    url: str

class PresetRequest(BaseModel):
    id: str = ""
    name: str
    args: str = ""
    description: str = ""

class LoadRequest(BaseModel):
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
    # Extra command line arguments
    extra_args: str = ""
    # IDs of presets currently toggled active in the UI. Persisted with model
    # settings so reloads restore the highlighted state. The preset args
    # themselves are already folded into `extra_args` by the frontend.
    active_preset_ids: list[str] = []
    # Speculative decoding draft model (MTP)
    draft_model_path: str = ""
    draft_gpu_layers: int = -1
    draft_max: int = 3
    draft_p_min: float = 0.0


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
    all_settings = load_all_settings()
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
            extra_args=req.extra_args,
            active_preset_ids=req.active_preset_ids,
            draft_model_path=req.draft_model_path,
            draft_gpu_layers=req.draft_gpu_layers,
            draft_max=req.draft_max,
            draft_p_min=req.draft_p_min,
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Unexpected error: {e}")

    return {"message": "Server starting…"}

@app.post("/api/server/stop")
async def server_stop():
    await server.stop()
    return {"message": "Server stopped"}


# ── Preset endpoints ─────────────────────────────────────

@app.get("/api/presets")
async def get_presets():
    return {"presets": presets.list_presets()}

@app.post("/api/presets")
async def upsert_preset(req: PresetRequest):
    try:
        record = presets.upsert_preset(req.model_dump())
    except ValueError as e:
        raise HTTPException(400, str(e))
    return record

@app.delete("/api/presets/{preset_id}")
async def delete_preset(preset_id: str):
    if not presets.delete_preset(preset_id):
        raise HTTPException(404, "Preset not found")
    return {"message": f"Deleted {preset_id}"}