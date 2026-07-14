#!/usr/bin/env python3
"""llama-cpp-webui — a lightweight GUI manager for llama.cpp"""

import argparse
import os
import sys

# Ensure this script's directory is importable (handles invocation from any cwd)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
import logging
from app import app


# Filter out the noisy polling requests from uvicorn access logs
class PollFilter(logging.Filter):
    SUPPRESSED = ("/api/build/status", "/api/models", "/api/server/status",
                  "/api/tabby/build/status", "/api/tabby/server/status", "/api/tabby/models")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(path in msg for path in self.SUPPRESSED)


def main():
    parser = argparse.ArgumentParser(description="llama-cpp-webui — llama.cpp Manager")
    parser.add_argument("--host", default="127.0.0.1", help="Management UI host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7800, help="Management UI port (default: 7800)")
    parser.add_argument("--data-dir", default=None, help="Directory for llama.cpp and models (default: <app-dir>/data)")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["LLAMA_CPP_WEBUI_DATA_DIR"] = args.data_dir

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║       llama-cpp-webui v1.0               ║")
    print(f"  ║   llama.cpp build & serve manager        ║")
    print(f"  ╚══════════════════════════════════════════╝\n")
    print(f"  → Management UI:  http://{args.host}:{args.port}")
    print(f"  → Inference API:  (shown after model load)\n")

    logging.getLogger("uvicorn.access").addFilter(PollFilter())
    # workers=1 is load-bearing: config.py's JSON persistence uses threading.Lock,
    # which is process-local. Multiple workers would race and lose settings updates.
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()