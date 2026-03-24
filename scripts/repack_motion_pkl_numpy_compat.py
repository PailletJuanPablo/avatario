"""
Repack one motion PKL using the NumPy version of the current interpreter.

Typical usage:
  python scripts/repack_motion_pkl_numpy_compat.py --input in.pkl --output out.pkl

When run under a NumPy 1.x interpreter, this converts pickles produced by NumPy 2.x
into a pickle that older environments can import.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path


def register_numpy_pickle_aliases() -> None:
    """
    Register compatibility aliases between NumPy 1.x and 2.x private module paths.
    """
    alias_pairs = (
        ("numpy._core.numeric", "numpy.core.numeric"),
        ("numpy._core.multiarray", "numpy.core.multiarray"),
        ("numpy.core.numeric", "numpy._core.numeric"),
        ("numpy.core.multiarray", "numpy._core.multiarray"),
    )
    for target_name, source_name in alias_pairs:
        if target_name in sys.modules:
            continue
        try:
            module = __import__(source_name, fromlist=["*"])
        except Exception:
            continue
        sys.modules[target_name] = module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repack one motion PKL for current NumPy compatibility.")
    parser.add_argument("--input", required=True, help="Input PKL path.")
    parser.add_argument("--output", required=True, help="Output PKL path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input PKL not found: {input_path}")

    register_numpy_pickle_aliases()
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[ok] repacked pkl -> {output_path}")


if __name__ == "__main__":
    main()
