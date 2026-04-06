"""Read metadata from GGUF file headers (no tensors loaded)."""
from __future__ import annotations

import struct
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# Value types
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32 = 0, 1, 2, 3, 4, 5
_FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_FMT = {
    _UINT8: ("<B", 1), _INT8: ("<b", 1),
    _UINT16: ("<H", 2), _INT16: ("<h", 2),
    _UINT32: ("<I", 4), _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _UINT64: ("<Q", 8), _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}


def _read_str(f) -> str:
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode("utf-8", errors="replace")


def _read_val(f, vtype: int):
    if vtype in _SCALAR_FMT:
        fmt, size = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, f.read(size))[0]
    if vtype == _BOOL:
        return struct.unpack("<B", f.read(1))[0] != 0
    if vtype == _STRING:
        return _read_str(f)
    if vtype == _ARRAY:
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        return [_read_val(f, arr_type) for _ in range(arr_len)]
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def _find(meta: dict, *suffixes):
    """Return first meta value whose key ends with any of the given suffixes."""
    for suffix in suffixes:
        v = meta.get(suffix)
        if v is not None:
            return v[0] if isinstance(v, list) else v
        for k, val in meta.items():
            if k.endswith("." + suffix):
                return val[0] if isinstance(val, list) else val
    return None


def read_metadata(path: str | Path) -> dict | None:
    """
    Parse a GGUF file header and return model architecture parameters.

    Returns dict with keys:
        n_layers      – transformer block count
        n_kv_heads    – attention KV head count
        embedding_dim – hidden/embedding dimension
        size_mb       – file size in MiB

    Returns None if the file is not a valid GGUF v2/v3, or if required
    fields cannot be found.
    """
    path = Path(path)
    size_mb = path.stat().st_size / (1024 * 1024)

    try:
        with open(path, "rb") as f:
            if f.read(4) != GGUF_MAGIC:
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                return None
            _tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            meta: dict = {}
            for _ in range(kv_count):
                key = _read_str(f)
                vtype = struct.unpack("<I", f.read(4))[0]
                meta[key] = _read_val(f, vtype)
    except Exception:
        return None

    n_layers = _find(meta, "block_count")
    # KV heads: prefer explicit count; fall back to full head count (MHA)
    n_kv_heads = _find(meta, "attention.head_count_kv", "attention.head_count")
    embedding_dim = _find(meta, "embedding_length")

    if not all(v is not None for v in (n_layers, n_kv_heads, embedding_dim)):
        return None
    # Some models store n_kv_heads = 0 meaning "same as n_heads"
    if n_kv_heads == 0:
        n_kv_heads = _find(meta, "attention.head_count")
    if not n_kv_heads:
        return None

    return {
        "n_layers": int(n_layers),
        "n_kv_heads": int(n_kv_heads),
        "embedding_dim": int(embedding_dim),
        "size_mb": size_mb,
    }
