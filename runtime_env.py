"""
Runtime environment helpers for local CUDA/TensorRT execution.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


WINDOWS_TRT_ALIAS_TARGETS: tuple[tuple[str, str], ...] = (
    ("nvinfer.dll", "nvinfer_10.dll"),
    ("nvinfer_plugin.dll", "nvinfer_plugin_10.dll"),
    ("nvinfer_dispatch.dll", "nvinfer_dispatch_10.dll"),
    ("nvinfer_lean.dll", "nvinfer_lean_10.dll"),
    ("nvinfer_vc_plugin.dll", "nvinfer_vc_plugin_10.dll"),
    ("nvinfer_builder_resource.dll", "nvinfer_builder_resource_10.dll"),
    ("nvonnxparser.dll", "nvonnxparser_10.dll"),
)
DEFAULT_WINDOWS_TENSORRT_LIB_DIRS: tuple[Path, ...] = (
    Path(r"C:\bin\tensorRT\lib"),
)


def _iter_unique_existing_dirs(candidates: list[Path]) -> list[Path]:
    """
    Normalize and deduplicate one list of directories while preserving order.
    """
    unique_paths: list[Path] = []
    seen_paths: set[str] = set()
    for candidate in candidates:
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            resolved_candidate = candidate
        if not resolved_candidate.exists() or not resolved_candidate.is_dir():
            continue
        normalized_key = str(resolved_candidate).lower()
        if normalized_key in seen_paths:
            continue
        seen_paths.add(normalized_key)
        unique_paths.append(resolved_candidate)
    return unique_paths


def resolve_windows_tensorrt_lib_dirs() -> list[Path]:
    """
    Resolve candidate TensorRT library directories for native Windows runs.
    """
    candidates: list[Path] = []
    env_lib_dir = str(os.getenv("TENSORRT_LIB_DIR", "")).strip()
    env_root_dir = str(os.getenv("TENSORRT_ROOT", "")).strip()
    if env_lib_dir:
        candidates.append(Path(env_lib_dir))
    if env_root_dir:
        env_root_path = Path(env_root_dir)
        candidates.append(env_root_path / "lib")
        candidates.append(env_root_path / "lib64")
    candidates.extend(DEFAULT_WINDOWS_TENSORRT_LIB_DIRS)
    return _iter_unique_existing_dirs(candidates)


def resolve_windows_cuda_bin_dirs() -> list[Path]:
    """
    Resolve CUDA binary directories that should be visible to TensorRT plugins.
    """
    candidates: list[Path] = []
    for env_key, env_value in os.environ.items():
        if env_key == "CUDA_PATH" or env_key.startswith("CUDA_PATH_V"):
            env_text = str(env_value or "").strip()
            if env_text:
                candidates.append(Path(env_text) / "bin")
    return _iter_unique_existing_dirs(candidates)


def ensure_windows_trt_alias_dir(project_root: Path, lib_dirs: list[Path]) -> Path | None:
    """
    Materialize generic TensorRT DLL aliases expected by some Windows plugins.
    """
    if not lib_dirs:
        return None
    alias_dir = project_root / "output_fasterliveportrait" / "trt_runtime_libs"
    alias_dir.mkdir(parents=True, exist_ok=True)
    wrote_any_alias = False
    for alias_name, target_name in WINDOWS_TRT_ALIAS_TARGETS:
        source_path = next((lib_dir / target_name for lib_dir in lib_dirs if (lib_dir / target_name).exists()), None)
        if source_path is None:
            continue
        alias_path = alias_dir / alias_name
        if alias_path.exists():
            try:
                same_size = alias_path.stat().st_size == source_path.stat().st_size
            except OSError:
                same_size = False
            if same_size:
                wrote_any_alias = True
                continue
        shutil.copy2(source_path, alias_path)
        wrote_any_alias = True
    return alias_dir if wrote_any_alias else None


def resolve_native_runtime_library_dirs(project_root: Path) -> list[Path]:
    """
    Resolve extra runtime library directories required for native Windows TRT execution.
    """
    if os.name != "nt":
        return []
    tensorrt_lib_dirs = resolve_windows_tensorrt_lib_dirs()
    runtime_dirs: list[Path] = []
    alias_dir = ensure_windows_trt_alias_dir(project_root, tensorrt_lib_dirs)
    if alias_dir is not None:
        runtime_dirs.append(alias_dir)
    runtime_dirs.extend(tensorrt_lib_dirs)
    runtime_dirs.extend(resolve_windows_cuda_bin_dirs())
    return _iter_unique_existing_dirs(runtime_dirs)


def _prepend_path_entries(runtime_env: dict[str, str], directories: list[Path]) -> None:
    """
    Prepend unique directories to PATH in one environment dictionary.
    """
    if not directories:
        return
    existing_entries = [entry for entry in runtime_env.get("PATH", "").split(os.pathsep) if entry]
    seen_entries = {entry.lower() for entry in existing_entries}
    prepended_entries: list[str] = []
    for directory in directories:
        path_text = str(directory)
        normalized_key = path_text.lower()
        if normalized_key in seen_entries:
            continue
        seen_entries.add(normalized_key)
        prepended_entries.append(path_text)
    if not prepended_entries:
        return
    runtime_env["PATH"] = os.pathsep.join(prepended_entries + existing_entries)


def build_process_env(project_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """
    Build one subprocess-safe environment with Windows TRT runtime paths when available.
    """
    runtime_env = dict(base_env or os.environ)
    runtime_env["PYTHONIOENCODING"] = "utf-8"
    runtime_env["PYTHONUTF8"] = "1"
    _prepend_path_entries(runtime_env, resolve_native_runtime_library_dirs(project_root))
    return runtime_env


def apply_runtime_library_environment(project_root: Path) -> None:
    """
    Update the current process environment in-place for native Windows TRT runs.
    """
    os.environ.update(build_process_env(project_root, os.environ))
