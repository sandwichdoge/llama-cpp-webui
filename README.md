# llama-cpp-webui

A lightweight management GUI for [llama.cpp](https://github.com/ggerganov/llama.cpp).
Builds the latest llama.cpp from source on demand, downloads GGUF models from
HuggingFace, and runs the OpenAI-compatible `llama-server` — all from a single
local web page.

## Features
- One-click clone / pull / rebuild of llama.cpp (auto-detects CUDA / Vulkan / CPU)
- HuggingFace `.gguf` and `mmproj` downloader with progress
- Model loader with per-model persisted settings
- Speculative decoding (MTP) with a draft model
- Global named preset bundles for extra CLI flags
- OpenAI-compatible inference endpoint at `http://localhost:<port>/v1`

## Start it

### Linux (Ubuntu / Debian)
```sh
./start_linux.sh
```

### macOS
```sh
./start_macos.sh
```

### Windows 10 / 11
Requires admin privileges to install missing dependencies:
```
run.bat
```

## CLI flags
`run.py` accepts:

| Flag         | Default     | Notes                                                        |
| ------------ | ----------- | ------------------------------------------------------------ |
| `--host`     | `127.0.0.1` | Management UI bind host                                      |
| `--port`     | `7800`      | Management UI port                                           |
| `--data-dir` | `./data`    | Where llama.cpp source, models, and settings are stored. Overrides `LLAMA_CPP_WEBUI_DATA_DIR`. |

The inference server (llama-server) defaults to binding on `127.0.0.1` for
safety. If you want it reachable from the LAN, add `--host 0.0.0.0` to the
"Extra command line arguments" field in the UI.

## Layout
```
app.py          FastAPI routes
server.py       llama-server lifecycle + LoadRequest schema
builder.py      git clone / pull / cmake build
downloader.py   HuggingFace GGUF downloads
gguf_reader.py  read GGUF header metadata (no tensors loaded)
presets.py      named CLI-flag bundles
config.py       paths + atomic JSON persistence
index.html      React single-page UI (Babel-in-browser)
run.py          uvicorn entry point
```
