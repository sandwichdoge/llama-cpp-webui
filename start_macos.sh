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
if ! command -v clang++ &>/dev/null && ! command -v g++ &>/dev/null; then
  missing+=("clang++ (Xcode CLT)")
fi

if [ ${#missing[@]} -gt 0 ]; then
  echo ""
  echo "  ⚠  Missing system dependencies: ${missing[*]}"
  echo ""
  echo "  Install Xcode Command Line Tools:"
  echo "    xcode-select --install"
  echo ""
  echo "  Install remaining tools with Homebrew:"
  echo "    brew install cmake python3"
  echo ""
  exit 1
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
