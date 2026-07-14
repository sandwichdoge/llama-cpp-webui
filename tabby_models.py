"""List, download, and delete exllamav3 model directories.

exl3 models are directories of safetensors (an HF repo snapshot), not single
files. Downloads are proxied through the running tabbyAPI's /v1/download.
"""

import asyncio
import re
import shutil
from pathlib import Path

import httpx

import tabby_server
from config import delete_exl3_settings, get_exl3_models_dir
from downloader import _human_size

# Active download state — keyed by repo id
_downloads: dict[str, dict] = {}

_REPO_ID_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
_REVISION_RE = re.compile(r"^[\w.-]*$")


def list_models() -> list[dict]:
    """List model directories with size info."""
    models = []
    for d in sorted(get_exl3_models_dir().iterdir()):
        if not d.is_dir():
            continue
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        models.append({
            "name": d.name,
            "path": str(d),
            "size_bytes": size,
            "size_human": _human_size(size),
            "valid": (d / "config.json").is_file() and any(d.glob("*.safetensors")),
        })
    return models


def get_downloads() -> dict[str, dict]:
    return dict(_downloads)


def resolve_model_dir(name: str) -> Path | None:
    """Return the canonical path for a model dir name. None on traversal."""
    if not name or name in (".", ".."):
        return None
    models_dir = get_exl3_models_dir().resolve()
    candidate = (models_dir / name).resolve()
    try:
        candidate.relative_to(models_dir)
    except ValueError:
        return None
    return candidate


def validate_download(repo_id: str, revision: str = "") -> tuple[str, str]:
    repo_id = repo_id.strip()
    if not _REPO_ID_RE.match(repo_id) or ".." in repo_id:
        raise ValueError("Repo id must look like 'user/repo', e.g. turboderp/Llama-3.2-1B-exl3")
    revision = revision.strip()
    if not _REVISION_RE.match(revision) or ".." in revision:
        raise ValueError("Revision must be a plain branch/tag name, e.g. 6.0bpw")
    return repo_id, revision


async def download(repo_id: str, revision: str = "") -> None:
    """Download an HF repo via tabby's /v1/download. Blocks until done —
    tabby streams no progress, so status is just downloading/complete/failed.

    exl3 quants conventionally live in branches (e.g. 6.0bpw), so a revision
    lands in its own folder to keep multiple quants of one repo apart.
    """
    if tabby_server.get_status()["status"] != "running":
        raise RuntimeError("Start tabbyAPI first — downloads go through its API.")

    body = {"repo_id": repo_id, "repo_type": "model"}
    folder_name = repo_id.split("/")[1]
    if revision and revision != "main":
        body["revision"] = revision
        folder_name = f"{folder_name}-{revision}"
        body["folder_name"] = folder_name

    # tabby's downloader errors on an existing folder — report it like
    # downloader.py does instead of surfacing a confusing failure.
    if (get_exl3_models_dir() / folder_name).exists():
        _downloads[repo_id] = {"status": "exists", "repo_id": repo_id,
                               "revision": revision, "error": None}
        return

    _downloads[repo_id] = {"status": "downloading", "repo_id": repo_id,
                           "revision": revision, "error": None}
    try:
        headers = tabby_server._admin_headers()
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{tabby_server._base_url()}/v1/download",
                                  json=body, headers=headers, timeout=None)
            if r.status_code != 200:
                raise RuntimeError(f"Download failed ({r.status_code}): {r.text[:500]}")
        _downloads[repo_id]["status"] = "complete"
    except asyncio.CancelledError:
        _downloads[repo_id]["status"] = "cancelled"
        raise
    except Exception as e:
        _downloads[repo_id]["status"] = "failed"
        _downloads[repo_id]["error"] = str(e)
        raise


def delete_model(name: str) -> bool:
    """Delete a model directory. Rejects traversal and the loaded model."""
    if name == tabby_server.loaded_model_name():
        raise RuntimeError("Model is currently loaded. Unload it first.")
    path = resolve_model_dir(name)
    if path and path.is_dir():
        shutil.rmtree(path)
        delete_exl3_settings(name)
        return True
    return False
