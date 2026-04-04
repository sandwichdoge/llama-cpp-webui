"""Paths and configuration for llama-cpp-webui."""

import os
import sys
from pathlib import Path


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