"""Paths and configuration for llama-cpp-webui."""

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger(__name__)

# Coarse per-file locks for the small JSON files we read/modify in place.
# Process-local — single-process assumption matches the rest of the app.
_settings_lock = threading.Lock()
_presets_lock = threading.Lock()


def get_data_dir() -> Path:
    default = Path(__file__).parent / "data"
    d = Path(os.environ.get("LLAMA_CPP_WEBUI_DATA_DIR", default))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_llama_cpp_dir() -> Path:
    return get_data_dir() / "llama.cpp"


def get_models_dir() -> Path:
    d = get_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_build_log_path() -> Path:
    return get_data_dir() / "build.log"


def get_server_binary() -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    build_bin = get_llama_cpp_dir() / "build" / "bin"
    # MSVC multi-config generators place the binary under a Release/ subdir;
    # Ninja / MinGW Makefiles use a flat layout.  Check both.
    for candidate in [
        build_bin / f"llama-server{suffix}",
        build_bin / "Release" / f"llama-server{suffix}",
    ]:
        if candidate.is_file():
            return candidate
    return build_bin / f"llama-server{suffix}"  # not yet built — return expected path


def get_settings_path() -> Path:
    return get_data_dir() / "model_settings.json"


def get_presets_path() -> Path:
    return get_data_dir() / "presets.json"


# ── JSON helpers ──────────────────────────────────────

def atomic_write_json(path: Path, data) -> None:
    """Write JSON atomically: write to a temp file, fsync, then rename.

    fsync on the temp file ensures its contents are on disk before the
    rename, so a crash either leaves the previous file intact or the new
    file fully written — never a half-written mix. The directory entry
    itself is not fsynced, so a crash *after* the rename may still revert
    to the previous file on some filesystems; acceptable for a local UI.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# Re-exported so presets.py can share the same primitive.
presets_lock = _presets_lock


# ── Settings persistence ──────────────────────────────

def _read_all_settings_locked() -> dict:
    """Read the settings file. Quarantine a corrupt JSON so the next save
    can't overwrite it with an empty dict (data-loss guard).
    Caller must hold _settings_lock.
    """
    p = get_settings_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup = p.with_name(f"{p.name}.corrupt.{int(time.time())}")
        try:
            p.rename(backup)
            log.error("model_settings.json was corrupt; quarantined to %s", backup.name)
        except OSError:
            log.exception("Failed to quarantine corrupt model_settings.json")
        return {}
    except OSError:
        return {}


def load_all_settings() -> dict:
    with _settings_lock:
        return _read_all_settings_locked()


def save_all_settings(data: dict) -> None:
    with _settings_lock:
        atomic_write_json(get_settings_path(), data)


def save_model_settings(filename: str, settings: dict) -> None:
    with _settings_lock:
        all_s = _read_all_settings_locked()
        all_s[filename] = settings
        atomic_write_json(get_settings_path(), all_s)


def delete_model_settings(filename: str) -> None:
    """Remove a model's settings entry, if present. No-op otherwise."""
    with _settings_lock:
        all_s = _read_all_settings_locked()
        if filename in all_s:
            del all_s[filename]
            atomic_write_json(get_settings_path(), all_s)
