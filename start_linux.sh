#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# ── Check system dependencies ────────────────────────────
missing=()
for cmd in python3 git cmake make; do
  if ! command -v "$cmd" &>/dev/null; then
    missing+=("$cmd")
  fi
done

# Check for C++ compiler
if ! command -v g++ &>/dev/null && ! command -v c++ &>/dev/null; then
  missing+=("g++")
fi

# Check python3-venv (required for venv creation)
if ! python3 -c "import venv" &>/dev/null; then
  missing+=("python3-venv")
fi

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo "  ⚠  Missing system dependencies: ${missing[*]}"
  echo ""
  echo "  Install them with:"
  echo "    sudo apt update && sudo apt install -y git cmake build-essential python3 python3-pip python3-venv"
  echo ""

  if command -v nvcc &>/dev/null; then
    echo "  ✓ CUDA detected — GPU acceleration will be enabled"
  elif [ -d "/opt/rocm" ]; then
    echo "  ✓ ROCm detected — GPU acceleration will be enabled"
  elif command -v vulkaninfo &>/dev/null; then
    echo "  ✓ Vulkan detected — GPU acceleration will be enabled"
  else
    echo "  ℹ  No GPU toolkit found (CUDA/ROCm/Vulkan) — will build CPU-only"
    echo "    For NVIDIA:  sudo apt install -y nvidia-cuda-toolkit"
    echo "    For Vulkan:  sudo apt install -y libvulkan-dev vulkan-tools"
  fi

  echo ""
  exit 1
fi

# ── Report GPU status on first run ───────────────────────
if [ ! -d ".venv" ]; then
  if command -v nvcc &>/dev/null; then
    echo "  ✓ CUDA detected"
  elif [ -d "/opt/rocm" ]; then
    echo "  ✓ ROCm detected"
  elif command -v vulkaninfo &>/dev/null; then
    echo "  ✓ Vulkan detected"
  else
    echo "  ℹ  No GPU toolkit — CPU-only build"
  fi
fi

# ── Create venv if needed ────────────────────────────────
if [ ! -d ".venv" ]; then
  echo "  Creating virtual environment…"
  python3 -m venv .venv
fi

source .venv/bin/activate

# Only re-install dependencies when requirements.txt changes (cheap mtime check).
marker=".venv/.deps-installed"
if [ ! -f "$marker" ] || [ requirements.txt -nt "$marker" ]; then
  pip install -q -r requirements.txt && touch "$marker"
fi

exec python run.py "$@"