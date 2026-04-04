"""Download GGUF models from HuggingFace."""

import asyncio
import re
import time
from pathlib import Path
from urllib.parse import urlparse, unquote

import httpx

from config import get_models_dir

# Active download state — keyed by filename
_downloads: dict[str, dict] = {}


def list_models() -> list[dict]:
    """List all downloaded GGUF files with size info."""
    models_dir = get_models_dir()
    models = []
    for f in sorted(models_dir.glob("*.gguf")):
        models.append({
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
            "size_human": _human_size(f.stat().st_size),
        })
    return models


def get_downloads() -> dict[str, dict]:
    """Return status of all active/finished downloads."""
    return dict(_downloads)


def parse_hf_url(url: str) -> tuple[str, str]:
    """Parse a HuggingFace URL into (download_url, filename).

    Supports formats:
      - https://huggingface.co/{user}/{repo}/resolve/main/{file}.gguf
      - https://huggingface.co/{user}/{repo}/blob/main/{file}.gguf  (auto-converts to resolve)
      - Direct CDN links (cdn-lfs.huggingface.co/...)
    """
    url = url.strip()

    # Convert blob URLs to resolve URLs
    url = url.replace("/blob/", "/resolve/")

    parsed = urlparse(url)
    filename = unquote(Path(parsed.path).name)

    if not filename.endswith(".gguf"):
        raise ValueError(f"URL does not point to a .gguf file: {filename}")

    return url, filename


async def download_model(url: str) -> str:
    """Download a GGUF model from a URL. Returns the filename."""
    download_url, filename = parse_hf_url(url)
    dest = get_models_dir() / filename

    if dest.exists():
        _downloads[filename] = {
            "status": "exists",
            "filename": filename,
            "progress": 100,
            "downloaded": dest.stat().st_size,
            "total": dest.stat().st_size,
            "speed": "",
            "error": None,
        }
        return filename

    _downloads[filename] = {
        "status": "downloading",
        "filename": filename,
        "progress": 0,
        "downloaded": 0,
        "total": 0,
        "speed": "",
        "error": None,
    }

    try:
        await _download_file(download_url, dest, filename)
        _downloads[filename]["status"] = "complete"
        _downloads[filename]["progress"] = 100
    except asyncio.CancelledError:
        _downloads[filename]["status"] = "cancelled"
        if dest.exists():
            dest.unlink()
        raise
    except Exception as e:
        _downloads[filename]["status"] = "failed"
        _downloads[filename]["error"] = str(e)
        # Clean up partial file
        if dest.exists():
            dest.unlink()
        raise

    return filename


async def _download_file(url: str, dest: Path, filename: str):
    """Download with progress tracking."""
    tmp = dest.with_suffix(".gguf.part")

    async with httpx.AsyncClient(follow_redirects=True, timeout=httpx.Timeout(30, read=300)) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            _downloads[filename]["total"] = total

            downloaded = 0
            last_time = time.monotonic()
            last_bytes = 0
            chunk_size = 1024 * 1024  # 1 MB

            with open(tmp, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.monotonic()
                    dt = now - last_time
                    if dt >= 0.5:  # Update speed every 0.5s
                        speed = (downloaded - last_bytes) / dt
                        _downloads[filename]["speed"] = _human_size(speed) + "/s"
                        last_time = now
                        last_bytes = downloaded

                    _downloads[filename]["downloaded"] = downloaded
                    if total > 0:
                        _downloads[filename]["progress"] = round(downloaded / total * 100, 1)

    tmp.rename(dest)


def delete_model(filename: str) -> bool:
    """Delete a downloaded model file."""
    path = get_models_dir() / filename
    if path.exists() and path.suffix == ".gguf":
        path.unlink()
        return True
    return False


def _human_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"
