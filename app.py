"""FastAPI management API for llama-cpp-webui."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import builder, downloader, gguf_reader, presets, server
import tabby_builder, tabby_models, tabby_server
from config import load_all_exl3_settings, load_all_settings
from server import LoadRequest
from tabby_server import TabbyLoadRequest, TabbyStartRequest

log = logging.getLogger("llama_cpp_webui")

presets.seed_if_empty()

_INDEX_HTML = Path(__file__).parent / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await server.shutdown()
    await tabby_server.shutdown()


app = FastAPI(title="llama-cpp-webui", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log unhandled errors and return a generic JSON 500 — never leak internals."""
    log.exception("Unhandled error in %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Request schemas ──────────────────────────────────────

class BuildRequest(BaseModel):
    gcc_flags: list[str] = Field(default_factory=list)

class DownloadRequest(BaseModel):
    url: str

class TabbyDownloadRequest(BaseModel):
    repo_id: str
    revision: str = ""

class PresetRequest(BaseModel):
    id: str = ""
    name: str
    args: str = ""
    description: str = ""


# ── UI ───────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse(_INDEX_HTML, media_type="text/html; charset=utf-8")


# ── Build endpoints ──────────────────────────────────────

@app.get("/api/build/status")
async def build_status():
    return builder.get_status()

@app.post("/api/build/start")
async def build_start(req: BuildRequest = BuildRequest()):
    if builder.is_building():
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
        await server.start(req)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"message": "Server starting…"}

@app.post("/api/server/stop")
async def server_stop():
    await server.stop()
    return {"message": "Server stopped"}


# ── tabbyAPI (exllamav3) endpoints ───────────────────────

@app.get("/api/tabby/build/status")
async def tabby_build_status():
    return tabby_builder.get_status()

@app.post("/api/tabby/build/start")
async def tabby_build_start():
    if tabby_builder.is_installing():
        raise HTTPException(409, "Install already in progress")
    asyncio.create_task(tabby_builder.install())
    return {"message": "Install started"}

@app.get("/api/tabby/server/status")
async def tabby_server_status():
    return tabby_server.get_status()

@app.post("/api/tabby/server/start")
async def tabby_server_start(req: TabbyStartRequest = TabbyStartRequest()):
    try:
        await tabby_server.start(req)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"message": "tabbyAPI starting…"}

@app.post("/api/tabby/server/stop")
async def tabby_server_stop():
    await tabby_server.stop()
    return {"message": "tabbyAPI stopped"}

@app.post("/api/tabby/server/load")
async def tabby_server_load(req: TabbyLoadRequest):
    try:
        await tabby_server.load_model(req)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(400, str(e))
    return {"message": "Model loading…"}

@app.post("/api/tabby/server/unload")
async def tabby_server_unload():
    try:
        await tabby_server.unload_model()
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"message": "Model unloaded"}

@app.get("/api/tabby/models")
async def tabby_list_models():
    all_settings = load_all_exl3_settings()
    models = tabby_models.list_models()
    for m in models:
        m["settings"] = all_settings.get(m["name"])
    return {
        "models": models,
        "models_dir": str(tabby_models.get_exl3_models_dir()),
        "downloads": tabby_models.get_downloads(),
    }

@app.post("/api/tabby/models/download")
async def tabby_download_model(req: TabbyDownloadRequest):
    try:
        repo_id, revision = tabby_models.validate_download(req.repo_id, req.revision)
        if tabby_server.get_status()["status"] != "running":
            raise RuntimeError("Start tabbyAPI first — downloads go through its API.")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    asyncio.create_task(tabby_models.download(repo_id, revision))
    return {"message": f"Download started: {repo_id}", "repo_id": repo_id}

@app.delete("/api/tabby/models/{name}")
async def tabby_delete_model(name: str):
    try:
        if not tabby_models.delete_model(name):
            raise HTTPException(404, "Model not found")
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return {"message": f"Deleted {name}"}


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