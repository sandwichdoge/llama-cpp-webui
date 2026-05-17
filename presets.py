"""Global, named bundles of extra llama-server CLI flags."""
from __future__ import annotations

import json
import re
import shlex
import uuid
from typing import Any

from config import get_presets_path

_BUILTIN_SEEDS: list[dict[str, str]] = [
    {
        "id": "mtp-type-1",
        "name": "Multi-token prediction (type 1)",
        "args": "--spec-type draft-mtp --spec-draft-n-max 3",
        "description": "Draft-MTP speculative decoding with n_max=3.",
    },
    {
        "id": "swa-gemma-32g",
        "name": "SWA Gemma — 32 GB RAM fit",
        "args": "-cram 4096",
        "description": "Caps sliding-window cache RAM so Gemma fits on a 32 GB host without OOM.",
    },
]


def _load_raw() -> dict[str, Any]:
    p = get_presets_path()
    if p.exists():
        try:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and isinstance(data.get("presets"), list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"presets": []}


def _save_raw(data: dict[str, Any]) -> None:
    get_presets_path().write_text(json.dumps(data, indent=2))


def seed_if_empty() -> None:
    """Populate the presets file with built-in examples if it doesn't exist."""
    if get_presets_path().exists():
        return
    _save_raw({"presets": list(_BUILTIN_SEEDS)})


def list_presets() -> list[dict[str, Any]]:
    return _load_raw()["presets"]


def get_preset(preset_id: str) -> dict[str, Any] | None:
    for p in list_presets():
        if p.get("id") == preset_id:
            return p
    return None


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or uuid.uuid4().hex[:8]


def upsert_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """Create or update a preset. Returns the stored record.

    Validates that `args` is shell-parseable so we fail at edit time, not launch time.
    """
    name = (preset.get("name") or "").strip()
    args = (preset.get("args") or "").strip()
    description = (preset.get("description") or "").strip()
    if not name:
        raise ValueError("Preset name is required")
    try:
        shlex.split(args)
    except ValueError as e:
        raise ValueError(f"Invalid args (shell-quoting): {e}")

    data = _load_raw()
    presets = data["presets"]
    preset_id = (preset.get("id") or "").strip() or _slugify(name)

    # Disambiguate auto-generated IDs against existing entries
    if not preset.get("id"):
        base, n = preset_id, 2
        existing = {p["id"] for p in presets}
        while preset_id in existing:
            preset_id = f"{base}-{n}"
            n += 1

    record = {"id": preset_id, "name": name, "args": args, "description": description}
    for i, p in enumerate(presets):
        if p.get("id") == preset_id:
            presets[i] = record
            break
    else:
        presets.append(record)
    _save_raw(data)
    return record


def delete_preset(preset_id: str) -> bool:
    data = _load_raw()
    before = len(data["presets"])
    data["presets"] = [p for p in data["presets"] if p.get("id") != preset_id]
    if len(data["presets"]) == before:
        return False
    _save_raw(data)
    return True
