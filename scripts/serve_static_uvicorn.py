"""
Serve project files over HTTP with proper media streaming support.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT_DIR = PROJECT_ROOT
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8787


def parse_args() -> argparse.Namespace:
    """
    Parse CLI options for static server.
    """
    parser = argparse.ArgumentParser(description="Serve static project files with Uvicorn.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--root", default=str(DEFAULT_ROOT_DIR))
    return parser.parse_args()


def resolve_root_dir(raw_root: str) -> Path:
    """
    Resolve root directory into absolute path.
    """
    candidate = Path(raw_root)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def create_app(root_dir: Path) -> FastAPI:
    """
    Build FastAPI app that serves static files from root.
    """
    app = FastAPI(title="Static Media Server")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
    )
    app.mount("/", StaticFiles(directory=str(root_dir), html=True), name="static")
    return app


def main() -> None:
    """
    Run static HTTP server.
    """
    args = parse_args()
    root_dir = resolve_root_dir(str(args.root))
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    app = create_app(root_dir=root_dir)
    uvicorn.run(app=app, host=str(args.host), port=int(args.port), log_level="info")


if __name__ == "__main__":
    main()
