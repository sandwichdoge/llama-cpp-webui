"""Paths and configuration for LlamaForge."""

import os
from pathlib import Path


def get_data_dir() -> Path:
    d = Path(os.environ.get("LLAMAFORGE_DATA_DIR", Path.home() / ".llamaforge"))
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
    return get_llama_cpp_dir() / "build" / "bin" / "llama-server"


def get_settings_path() -> Path:
    return get_data_dir() / "model_settings.json"