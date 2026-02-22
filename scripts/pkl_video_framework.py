"""
Minimal PKL-first framework for offline talking-video generation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKER_CONTAINER = "animation_api"
DEFAULT_DOCKER_PYTHON = "/root/miniconda3/bin/python"
DEFAULT_FASTER_REPO = "third_party/FasterLivePortrait"
DEFAULT_CFG = "third_party/FasterLivePortrait/configs/trt_infer.yaml"
DEFAULT_SOURCE_IMAGE = "output/frames/frame_00095.png"
RUNTIME_DOCKER = "docker"
RUNTIME_LOCAL = "local"


def resolve_path(path_value: str) -> Path:
    """
    Resolve path relative to project root.
    """
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def to_container_path(path_value: Path) -> str:
    """
    Convert project path to /app path inside animation_api container.
    """
    relative = path_value.resolve().relative_to(PROJECT_ROOT.resolve())
    return f"/app/{relative.as_posix()}"


def run_command(command: list[str]) -> None:
    """
    Execute command and fail fast on non-zero exit code.
    """
    print("\n[cmd] " + " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    """
    Parse subcommands.
    """
    parser = argparse.ArgumentParser(description="PKL-first mini framework.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_from_audio = subparsers.add_parser("build-pkl", help="Build one pkl from one audio.")
    build_from_audio.add_argument("--audio", required=True)
    build_from_audio.add_argument("--output-pkl", required=True)
    build_from_audio.add_argument("--seed", type=int, default=1234)
    build_from_audio.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO)
    build_from_audio.add_argument("--cfg", default=DEFAULT_CFG)
    build_from_audio.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    build_from_audio.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)

    tune = subparsers.add_parser("tune-pkl", help="Apply eye-tamed tuning to one pkl.")
    tune.add_argument("--input-pkl", required=True)
    tune.add_argument("--output-pkl", required=True)
    tune.add_argument("--runtime", choices=[RUNTIME_DOCKER, RUNTIME_LOCAL], default=RUNTIME_DOCKER)
    tune.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    tune.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    tune.add_argument("--soft-factor", type=float, default=0.45)
    tune.add_argument("--hard-factor", type=float, default=0.18)
    tune.add_argument("--hard-dy-min", type=float, default=-0.0045)
    tune.add_argument("--hard-dy-max", type=float, default=0.0035)

    render = subparsers.add_parser("render-pkl", help="Render video from one pkl.")
    render.add_argument("--pkl", required=True)
    render.add_argument("--output-dir", required=True)
    render.add_argument("--source-image", default=DEFAULT_SOURCE_IMAGE)
    render.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO)
    render.add_argument("--cfg", default=DEFAULT_CFG)
    render.add_argument("--source-cache-dir", default="output_fasterliveportrait/source_preprocess_cache/pkl_framework")
    render.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    render.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    render.add_argument("--paste-back", action="store_true", default=True)
    render.add_argument("--no-paste-back", dest="paste_back", action="store_false")

    pipeline = subparsers.add_parser(
        "build-tune-render",
        help="Build pkl from audio, tune eyes, and render one output video.",
    )
    pipeline.add_argument("--audio", required=True)
    pipeline.add_argument("--work-dir", required=True)
    pipeline.add_argument("--source-image", default=DEFAULT_SOURCE_IMAGE)
    pipeline.add_argument("--seed", type=int, default=1234)
    pipeline.add_argument("--faster-repo-dir", default=DEFAULT_FASTER_REPO)
    pipeline.add_argument("--cfg", default=DEFAULT_CFG)
    pipeline.add_argument("--source-cache-dir", default="output_fasterliveportrait/source_preprocess_cache/pkl_framework")
    pipeline.add_argument("--docker-container", default=DEFAULT_DOCKER_CONTAINER)
    pipeline.add_argument("--docker-python", default=DEFAULT_DOCKER_PYTHON)
    pipeline.add_argument("--paste-back", action="store_true", default=True)
    pipeline.add_argument("--no-paste-back", dest="paste_back", action="store_false")
    return parser.parse_args()


def run_build_pkl(args: argparse.Namespace) -> Path:
    """
    Build one pkl from one driving audio.
    """
    audio_path = resolve_path(args.audio)
    output_pkl = resolve_path(args.output_pkl)
    faster_repo = resolve_path(args.faster_repo_dir)
    cfg_path = resolve_path(args.cfg)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "exec",
        str(args.docker_container),
        str(args.docker_python),
        to_container_path(resolve_path("faster_liveportrait_audio_to_pkl.py")),
        "--faster-repo-dir",
        to_container_path(faster_repo),
        "--cfg",
        to_container_path(cfg_path),
        "--driving-audio",
        to_container_path(audio_path),
        "--output-pkl",
        to_container_path(output_pkl),
        "--seed",
        str(int(args.seed)),
    ]
    run_command(command)
    return output_pkl


def run_tune_pkl(args: argparse.Namespace) -> Path:
    """
    Tune one pkl with eye-tamed profile.
    """
    input_pkl = resolve_path(args.input_pkl)
    output_pkl = resolve_path(args.output_pkl)
    if not input_pkl.exists():
        raise FileNotFoundError(f"Input pkl not found: {input_pkl}")
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    if str(args.runtime) == RUNTIME_DOCKER:
        tune_inline = (
            "import pickle, sys, numpy as np\n"
            "src, dst = sys.argv[1], sys.argv[2]\n"
            "soft = float(sys.argv[3])\n"
            "hard = float(sys.argv[4])\n"
            "dymin = float(sys.argv[5])\n"
            "dymax = float(sys.argv[6])\n"
            "soft_idx = (0,1,2,3,4,5,7,10,13)\n"
            "hard_idx = (11,15)\n"
            "payload = pickle.load(open(src,'rb'))\n"
            "motion = payload.get('motion', [])\n"
            "assert isinstance(motion, list) and len(motion) > 0, 'invalid motion payload'\n"
            "base = np.asarray(motion[0]['exp'], dtype=np.float32).reshape(21,3).copy()\n"
            "safe_soft = float(np.clip(soft, 0.0, 1.0))\n"
            "safe_hard = float(np.clip(hard, 0.0, 1.0))\n"
            "mn = float(min(dymin, dymax))\n"
            "mx = float(max(dymin, dymax))\n"
            "out = []\n"
            "for frame in motion:\n"
            "    exp = np.asarray(frame['exp'], dtype=np.float32).reshape(21,3).copy()\n"
            "    for idx in soft_idx:\n"
            "        exp[idx,:] = base[idx,:] + (exp[idx,:] - base[idx,:]) * safe_soft\n"
            "    for idx in hard_idx:\n"
            "        exp[idx,:] = base[idx,:] + (exp[idx,:] - base[idx,:]) * safe_hard\n"
            "        dy = exp[idx,1] - base[idx,1]\n"
            "        exp[idx,1] = base[idx,1] + float(np.clip(dy, mn, mx))\n"
            "    next_frame = dict(frame)\n"
            "    next_frame['exp'] = exp.reshape(1,21,3)\n"
            "    out.append(next_frame)\n"
            "payload['motion'] = out\n"
            "pickle.dump(payload, open(dst,'wb'))\n"
            "print('[ok] tuned pkl ->', dst)\n"
        )
        command = [
            "docker",
            "exec",
            str(args.docker_container),
            str(args.docker_python),
            "-c",
            tune_inline,
            to_container_path(input_pkl),
            to_container_path(output_pkl),
            str(float(args.soft_factor)),
            str(float(args.hard_factor)),
            str(float(args.hard_dy_min)),
            str(float(args.hard_dy_max)),
        ]
    else:
        command = [
            sys.executable,
            str(resolve_path("scripts/tune_motion_pkl.py")),
            "--input-pkl",
            str(input_pkl),
            "--output-pkl",
            str(output_pkl),
            "--soft-factor",
            str(float(args.soft_factor)),
            "--hard-factor",
            str(float(args.hard_factor)),
            "--hard-dy-min",
            str(float(args.hard_dy_min)),
            "--hard-dy-max",
            str(float(args.hard_dy_max)),
        ]
    run_command(command)
    return output_pkl


def run_render_pkl(args: argparse.Namespace) -> Path:
    """
    Render one pkl to output video folder.
    """
    pkl_path = resolve_path(args.pkl)
    output_dir = resolve_path(args.output_dir)
    source_image = resolve_path(args.source_image)
    faster_repo = resolve_path(args.faster_repo_dir)
    cfg_path = resolve_path(args.cfg)
    source_cache_dir = resolve_path(args.source_cache_dir)
    if not pkl_path.exists():
        raise FileNotFoundError(f"PKL not found: {pkl_path}")
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "exec",
        "-w",
        to_container_path(faster_repo),
        str(args.docker_container),
        str(args.docker_python),
        "run.py",
        "--src_image",
        to_container_path(source_image),
        "--dri_video",
        to_container_path(pkl_path),
        "--cfg",
        to_container_path(cfg_path),
        "--source_cache_dir",
        to_container_path(source_cache_dir),
        "--save_dir",
        to_container_path(output_dir),
    ]
    if bool(args.paste_back):
        command.append("--paste_back")
    run_command(command)
    print(f"[ok] rendered into {output_dir}")
    return output_dir


def run_build_tune_render(args: argparse.Namespace) -> None:
    """
    End-to-end helper pipeline.
    """
    work_dir = resolve_path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_pkl = work_dir / "motion_raw.pkl"
    tuned_pkl = work_dir / "motion_eye_tamed.pkl"
    render_dir = work_dir / "render"

    build_args = argparse.Namespace(**vars(args))
    build_args.output_pkl = str(raw_pkl)
    run_build_pkl(build_args)

    tune_args = argparse.Namespace(
        input_pkl=str(raw_pkl),
        output_pkl=str(tuned_pkl),
        runtime=RUNTIME_DOCKER,
        docker_container=str(args.docker_container),
        docker_python=str(args.docker_python),
        soft_factor=0.45,
        hard_factor=0.18,
        hard_dy_min=-0.0045,
        hard_dy_max=0.0035,
    )
    run_tune_pkl(tune_args)

    render_args = argparse.Namespace(**vars(args))
    render_args.pkl = str(tuned_pkl)
    render_args.output_dir = str(render_dir)
    run_render_pkl(render_args)

    print(f"[ok] raw pkl   -> {raw_pkl}")
    print(f"[ok] tuned pkl -> {tuned_pkl}")
    print(f"[ok] render    -> {render_dir}")


def main() -> None:
    """
    Program entry point.
    """
    args = parse_args()
    if args.command == "build-pkl":
        path = run_build_pkl(args)
        print(f"[ok] pkl -> {path}")
        return
    if args.command == "tune-pkl":
        path = run_tune_pkl(args)
        print(f"[ok] tuned -> {path}")
        return
    if args.command == "render-pkl":
        path = run_render_pkl(args)
        print(f"[ok] output dir -> {path}")
        return
    if args.command == "build-tune-render":
        run_build_tune_render(args)
        return
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
