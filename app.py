"""FastAPI management API for LlamaForge."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import builder, downloader, server
from config import get_settings_path

app = FastAPI(title="LlamaForge", version="1.0.0")


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

class DownloadRequest(BaseModel):
    url: str

class LoadRequest(BaseModel):
    model_path: str
    port: int = 8080
    n_gpu_layers: int = -1
    ctx_size: int = 4096
    n_parallel: int = 1


# ── UI ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())


# ── Build endpoints ──────────────────────────────────────

@app.get("/api/build/status")
async def build_status():
    return builder.get_status()

@app.post("/api/build/start")
async def build_start():
    status = builder.get_status()
    if status["status"] == "building":
        raise HTTPException(409, "Build already in progress")
    asyncio.create_task(builder.build())
    return {"message": "Build started"}


# ── Model endpoints ──────────────────────────────────────

@app.get("/api/models")
async def list_models():
    all_settings = _load_all_settings()
    models = downloader.list_models()
    for m in models:
        m["settings"] = all_settings.get(m["filename"])
    return {
        "models": models,
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
        )
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))

    # Persist settings for this model
    filename = Path(req.model_path).name
    _save_model_settings(filename, {
        "port": req.port,
        "n_gpu_layers": req.n_gpu_layers,
        "ctx_size": req.ctx_size,
        "n_parallel": req.n_parallel,
    })

    return {"message": "Server starting…"}

@app.post("/api/server/stop")
async def server_stop():
    await server.stop()
    return {"message": "Server stopped"}