#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
STATE_ROOT="${PROJECT_ROOT}/.runpod"
TOKEN_FILE_PATH="${STATE_ROOT}/api_token"
LOG_FILE_PATH="${PROJECT_ROOT}/output_fasterliveportrait/runpod_api.log"
DEFAULT_PYTHON_BIN="/root/miniconda3/bin/python"
DEFAULT_PIP_BIN="/root/miniconda3/bin/pip"
DEFAULT_API_PORT="8010"
DEFAULT_API_HOST="0.0.0.0"
DEFAULT_BACKEND="trt"
DEFAULT_TRT_RUNTIME="local"
DEFAULT_TRT_PRECISION="fp16"
DEFAULT_IDLE_VIDEO_PATH="inputs/idlevid.mp4"
DEFAULT_CHECKPOINT_REPO_ID="warmshao/FasterLivePortrait"
DEFAULT_JOYVASA_CHECKPOINT_REPO_ID="jdh-algo/JoyVASA"
DEFAULT_CHINESE_HUBERT_REPO_ID="TencentGameMate/chinese-hubert-base"
DEFAULT_IMAGE_BUNDLE_DIR="/app/runpod-bundle"
RUNPOD_REQUIREMENTS_FILE_PATH="${PROJECT_ROOT}/requirements-runpod.txt"
DEFAULT_TRT_PLUGIN_LIBRARY_PATH="${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so"
DEFAULT_SYSTEM_PACKAGES=(
  "ca-certificates"
  "curl"
  "ffmpeg"
  "git"
  "libglib2.0-0"
  "libsm6"
  "libxext6"
  "libxrender1"
)
HEALTHCHECK_MAX_ATTEMPTS="${RUNPOD_HEALTHCHECK_MAX_ATTEMPTS:-180}"
HEALTHCHECK_SLEEP_SECONDS="${RUNPOD_HEALTHCHECK_SLEEP_SECONDS:-2}"

API_PID=""
PYTHON_BIN=""
PIP_BIN=""

print_info() {
  printf '[info] %s\n' "$1"
}

print_warning() {
  printf '[warn] %s\n' "$1" >&2
}

print_error() {
  printf '[error] %s\n' "$1" >&2
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

apt_package_installed() {
  local package_name="$1"
  if command_exists dpkg-query; then
    dpkg-query -W -f='${Status}' "${package_name}" 2>/dev/null | grep -q "install ok installed"
    return
  fi
  return 1
}

install_apt_packages_if_missing() {
  local package_names=("$@")
  local packages_to_install=()
  local package_name
  local binary_name

  if ! command_exists apt-get; then
    if ((${#package_names[@]} > 0)); then
      print_error "apt-get is unavailable; cannot install missing system packages: ${package_names[*]}"
    fi
    return 1
  fi

  for package_name in "${package_names[@]}"; do
    if apt_package_installed "${package_name}"; then
      continue
    fi
    binary_name="${package_name}"
    case "${package_name}" in
      ca-certificates) binary_name="update-ca-certificates" ;;
      openssh-server) binary_name="sshd" ;;
      python3-pip) binary_name="pip3" ;;
    esac
    if command_exists "${binary_name}"; then
      continue
    fi
    packages_to_install+=("${package_name}")
  done

  if ((${#packages_to_install[@]} == 0)); then
    return 0
  fi

  print_info "Installing system packages: ${packages_to_install[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends "${packages_to_install[@]}"
}

resolve_python_bin() {
  if [[ -n "${RUNPOD_PYTHON_BIN:-}" && -x "${RUNPOD_PYTHON_BIN}" ]]; then
    printf '%s\n' "${RUNPOD_PYTHON_BIN}"
    return
  fi
  if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
    printf '%s\n' "${DEFAULT_PYTHON_BIN}"
    return
  fi
  if command_exists python3; then
    command -v python3
    return
  fi
  if command_exists python; then
    command -v python
    return
  fi
  print_error "Python runtime not found. Use an official Runpod PyTorch image or a custom image built from Dockerfile.runpod."
  exit 1
}

resolve_pip_bin() {
  local python_bin="$1"
  local python_dir
  python_dir="$(cd "$(dirname "${python_bin}")" && pwd)"
  if [[ -n "${RUNPOD_PIP_BIN:-}" && -x "${RUNPOD_PIP_BIN}" ]]; then
    printf '%s\n' "${RUNPOD_PIP_BIN}"
    return
  fi
  if [[ -x "${python_dir}/pip" ]]; then
    printf '%s\n' "${python_dir}/pip"
    return
  fi
  if [[ -x "${DEFAULT_PIP_BIN}" ]]; then
    printf '%s\n' "${DEFAULT_PIP_BIN}"
    return
  fi
  if command_exists pip3; then
    command -v pip3
    return
  fi
  if command_exists pip; then
    command -v pip
    return
  fi
  print_error "pip runtime not found. Use an official Runpod PyTorch image or a custom image built from Dockerfile.runpod."
  exit 1
}

ensure_runpod_system_dependencies() {
  install_apt_packages_if_missing "${DEFAULT_SYSTEM_PACKAGES[@]}" >/dev/null
}

find_latest_nvidia_library_candidate() {
  local library_glob_patterns=("$@")
  local pattern
  local candidate_path
  local candidate_paths=()

  for pattern in "${library_glob_patterns[@]}"; do
    while IFS= read -r candidate_path; do
      if [[ -n "${candidate_path}" ]]; then
        candidate_paths+=("${candidate_path}")
      fi
    done < <(compgen -G "${pattern}" || true)
  done

  if ((${#candidate_paths[@]} == 0)); then
    return
  fi

  printf '%s\n' "${candidate_paths[@]}" | sort -V | tail -n 1
}

repair_driver_library_link_if_needed() {
  local link_path="$1"
  shift
  local target_path=""

  if [[ -L "${link_path}" && -s "${link_path}" ]]; then
    return
  fi
  if [[ -f "${link_path}" && -s "${link_path}" ]]; then
    return
  fi

  target_path="$(find_latest_nvidia_library_candidate "$@")"
  if [[ -z "${target_path}" ]]; then
    print_warning "Could not repair ${link_path}; no candidate library was found."
    return
  fi

  rm -f "${link_path}"
  ln -s "${target_path}" "${link_path}"
  print_info "Repaired NVIDIA driver link: ${link_path} -> ${target_path}"
}

repair_nvidia_driver_links_if_needed() {
  local ldconfig_bin=""

  repair_driver_library_link_if_needed \
    "/lib/x86_64-linux-gnu/libcuda.so.1" \
    "/lib/x86_64-linux-gnu/libcuda.so.[0-9]*.[0-9]*" \
    "/usr/lib/x86_64-linux-gnu/libcuda.so.[0-9]*.[0-9]*"

  if [[ ! -L "/lib/x86_64-linux-gnu/libcuda.so" || ! -s "/lib/x86_64-linux-gnu/libcuda.so" ]]; then
    rm -f "/lib/x86_64-linux-gnu/libcuda.so"
    ln -s "/lib/x86_64-linux-gnu/libcuda.so.1" "/lib/x86_64-linux-gnu/libcuda.so"
    print_info "Repaired NVIDIA driver link: /lib/x86_64-linux-gnu/libcuda.so -> /lib/x86_64-linux-gnu/libcuda.so.1"
  fi

  repair_driver_library_link_if_needed \
    "/lib/x86_64-linux-gnu/libnvidia-ml.so.1" \
    "/lib/x86_64-linux-gnu/libnvidia-ml.so.[0-9]*.[0-9]*" \
    "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.[0-9]*.[0-9]*"

  if command_exists ldconfig; then
    ldconfig_bin="$(command -v ldconfig)"
    "${ldconfig_bin}" || true
  fi
}

ensure_gpu_runtime() {
  local python_bin="$1"
  local cuda_available

  if command_exists nvidia-smi; then
    print_info "GPU runtime:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  else
    print_warning "nvidia-smi is unavailable; validating CUDA access through PyTorch."
  fi

  cuda_available="$("${python_bin}" - <<'PY'
import torch
print("1" if torch.cuda.is_available() else "0")
PY
)"
  if [[ "${cuda_available}" != "1" ]]; then
    print_error "CUDA is unavailable in this Pod. Terminate the Pod instead of continuing."
    exit 1
  fi
}

ensure_trt_builder_runtime() {
  local python_bin="$1"
  local runtime_check_output

  runtime_check_output="$("${python_bin}" - <<'PY'
import sys

import torch

try:
    import pycuda.autoinit  # noqa: F401
    import tensorrt as trt
except Exception as exc:
    print(f"import_error={exc!r}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("torch_cuda_available=False")
    sys.exit(1)

try:
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
except Exception as exc:
    print(f"builder_error={exc!r}")
    sys.exit(1)

if builder is None:
    print("builder_error=Builder returned None")
    sys.exit(1)

print("trt_builder_ready=True")
PY
)" || {
    print_error "TensorRT runtime self-test failed."
    print_error "${runtime_check_output}"
    print_error "Terminate the Pod instead of continuing."
    exit 1
  }

  print_info "${runtime_check_output}"
}

ensure_trt_plugin_runtime() {
  local python_bin="$1"
  local runtime_check_output

  if [[ ! -f "${DEFAULT_TRT_PLUGIN_LIBRARY_PATH}" ]]; then
    print_error "TensorRT plugin library not found: ${DEFAULT_TRT_PLUGIN_LIBRARY_PATH}"
    exit 1
  fi

  runtime_check_output="$("${python_bin}" - <<PY
import ctypes
import os
import sys

plugin_path = os.path.abspath(${DEFAULT_TRT_PLUGIN_LIBRARY_PATH@Q})

try:
    ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
except Exception as exc:
    print(f"plugin_error={exc!r}")
    sys.exit(1)

print("trt_plugin_ready=True")
PY
)" || {
    print_error "TensorRT plugin self-test failed."
    print_error "${runtime_check_output}"
    print_error "Terminate the Pod instead of continuing."
    exit 1
  }

  print_info "${runtime_check_output}"
}

ensure_python_runtime_modules() {
  local python_bin="$1"
  local pip_bin="$2"
  local missing_preinstalled_modules
  local missing_runtime_modules

  missing_preinstalled_modules="$("${python_bin}" - <<'PY'
import importlib.util

required_preinstalled_modules = ("numpy", "torch")
missing_modules = [name for name in required_preinstalled_modules if importlib.util.find_spec(name) is None]
print(" ".join(missing_modules))
PY
)"

  if [[ -n "${missing_preinstalled_modules}" ]]; then
    print_error "The selected image is missing required CUDA runtime modules: ${missing_preinstalled_modules}"
    print_error "Use a Runpod PyTorch image with CUDA support."
    exit 1
  fi

  if [[ ! -f "${RUNPOD_REQUIREMENTS_FILE_PATH}" ]]; then
    print_error "Runpod requirements file not found: ${RUNPOD_REQUIREMENTS_FILE_PATH}"
    exit 1
  fi

  print_info "Installing Python runtime packages from ${RUNPOD_REQUIREMENTS_FILE_PATH}"
  "${pip_bin}" install --no-cache-dir -r "${RUNPOD_REQUIREMENTS_FILE_PATH}"

  missing_runtime_modules="$("${python_bin}" - <<'PY'
import importlib.util

required_runtime_modules = (
    "aiortc",
    "av",
    "colorama",
    "cv2",
    "fastapi",
    "ffmpeg",
    "huggingface_hub",
    "insightface",
    "mediapipe",
    "multipart",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "PIL",
    "pycuda",
    "skimage",
    "torchgeometry",
    "tqdm",
    "transformers",
    "uvicorn",
)
missing_modules = [name for name in required_runtime_modules if importlib.util.find_spec(name) is None]
print(" ".join(missing_modules))
PY
)"

  if [[ -n "${missing_runtime_modules}" ]]; then
    print_error "Missing Python runtime modules after bootstrap: ${missing_runtime_modules}"
    exit 1
  fi

  if ! "${python_bin}" - <<'PY'
import importlib.util
import sys

torchaudio_spec = importlib.util.find_spec("torchaudio")
sys.exit(0 if torchaudio_spec is not None else 1)
PY
  then
    print_error "torchaudio is unavailable in the selected image. Use an official Runpod PyTorch image that includes torchaudio."
    exit 1
  fi
}

ensure_project_layout() {
  mkdir -p \
    "${PROJECT_ROOT}/inputs" \
    "${PROJECT_ROOT}/output" \
    "${PROJECT_ROOT}/output_fasterliveportrait" \
    "${PROJECT_ROOT}/third_party" \
    "${STATE_ROOT}"
}

ensure_faster_liveportrait_repo() {
  local faster_liveportrait_dir="${PROJECT_ROOT}/third_party/FasterLivePortrait"
  if [[ -f "${faster_liveportrait_dir}/run.py" ]]; then
    return
  fi
  if [[ -e "${faster_liveportrait_dir}" ]]; then
    print_warning "Removing incomplete FasterLivePortrait directory: ${faster_liveportrait_dir}"
    rm -rf "${faster_liveportrait_dir}"
  fi
  print_info "Bootstrapping FasterLivePortrait dependency"
  bash "${PROJECT_ROOT}/scripts/bootstrap_faster_liveportrait.sh"
  if [[ ! -f "${faster_liveportrait_dir}/run.py" ]]; then
    print_error "FasterLivePortrait bootstrap did not produce run.py: ${faster_liveportrait_dir}"
    exit 1
  fi
}

apply_faster_liveportrait_overrides_if_present() {
  local bundle_dir="${RUNPOD_IMAGE_BUNDLE_DIR:-${DEFAULT_IMAGE_BUNDLE_DIR}}"
  local override_dir="${bundle_dir}/faster_liveportrait_overrides"
  local target_dir="${PROJECT_ROOT}/third_party/FasterLivePortrait"

  if [[ ! -d "${override_dir}" ]]; then
    return
  fi
  if [[ ! -d "${target_dir}" ]]; then
    print_error "FasterLivePortrait repo missing before applying overrides: ${target_dir}"
    exit 1
  fi

  print_info "Applying bundled FasterLivePortrait overrides"
  cp -a "${override_dir}/." "${target_dir}/"
  chmod +x "${target_dir}/run_persistent_worker.py" >/dev/null 2>&1 || true
}

ensure_idle_video_exists() {
  local idle_video_path="${ANIMATION_IDLE_VIDEO}"
  if [[ "${idle_video_path}" != /* ]]; then
    idle_video_path="${PROJECT_ROOT}/${idle_video_path}"
  fi
  if [[ -f "${idle_video_path}" ]]; then
    return
  fi
  print_error "Idle video not found: ${idle_video_path}"
  exit 1
}

resolve_idle_video_abs_path() {
  local idle_video_path="${ANIMATION_IDLE_VIDEO}"
  if [[ "${idle_video_path}" != /* ]]; then
    idle_video_path="${PROJECT_ROOT}/${idle_video_path}"
  fi
  printf '%s\n' "${idle_video_path}"
}

write_source_meta_from_idle_video() {
  local idle_video_path="$1"
  local meta_path="$2"

  "${PYTHON_BIN}" - "${idle_video_path}" "${meta_path}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

idle_video_path = Path(sys.argv[1]).resolve()
meta_path = Path(sys.argv[2]).resolve()

capture = cv2.VideoCapture(str(idle_video_path))
if not capture.isOpened():
    raise SystemExit(f"Unable to open idle video for metadata generation: {idle_video_path}")

fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
capture.release()

if fps <= 0:
    fps = 25.0

meta_path.parent.mkdir(parents=True, exist_ok=True)
meta_path.write_text(
    json.dumps(
        {
            "fps": fps,
            "frameCount": frame_count,
            "sourceVideo": str(idle_video_path),
            "generatedBy": "scripts/runpod_bootstrap.sh",
        },
        indent=2,
    ),
    encoding="utf-8",
)
PY
}

extract_source_frames_from_idle_video() {
  local idle_video_path="$1"
  local frames_dir_path="$2"

  find "${frames_dir_path}" -maxdepth 1 -type f -name 'frame_*.png' -delete
  ffmpeg -hide_banner -loglevel error -y \
    -i "${idle_video_path}" \
    -start_number 1 \
    "${frames_dir_path}/frame_%05d.png"
}

ensure_source_assets_exist() {
  local frames_dir_path="${PROJECT_ROOT}/output/frames"
  local meta_path="${PROJECT_ROOT}/output/meta.json"
  local idle_video_path
  local needs_frames="0"
  local needs_meta="0"
  local force_regeneration="${RUNPOD_REGENERATE_SOURCE_FRAMES:-0}"
  local first_frame_path=""

  mkdir -p "${frames_dir_path}"
  idle_video_path="$(resolve_idle_video_abs_path)"

  first_frame_path="$(find "${frames_dir_path}" -maxdepth 1 -type f -name 'frame_*.png' -print -quit)"
  if [[ -z "${first_frame_path}" ]]; then
    needs_frames="1"
  fi
  if [[ ! -f "${meta_path}" ]]; then
    needs_meta="1"
  fi

  if [[ "${needs_frames}" == "0" && "${needs_meta}" == "0" && "${force_regeneration}" != "1" ]]; then
    return
  fi

  print_info "Generating source assets from idle video: ${idle_video_path}"

  if [[ "${needs_frames}" == "1" || "${force_regeneration}" == "1" ]]; then
    extract_source_frames_from_idle_video "${idle_video_path}" "${frames_dir_path}"
  fi

  if [[ "${needs_meta}" == "1" || "${force_regeneration}" == "1" ]]; then
    write_source_meta_from_idle_video "${idle_video_path}" "${meta_path}"
  fi

  first_frame_path="$(find "${frames_dir_path}" -maxdepth 1 -type f -name 'frame_*.png' -print -quit)"
  if [[ -z "${first_frame_path}" ]]; then
    print_error "Source frame generation failed. No frame_*.png files exist in ${frames_dir_path}"
    exit 1
  fi
  if [[ ! -f "${meta_path}" ]]; then
    print_error "Source metadata generation failed: ${meta_path}"
    exit 1
  fi
}

resolve_huggingface_cli_bin() {
  local python_bin="$1"
  local python_dir
  python_dir="$(cd "$(dirname "${python_bin}")" && pwd)"
  if [[ -x "${python_dir}/hf" ]]; then
    printf '%s\n' "${python_dir}/hf"
    return
  fi
  if command_exists hf; then
    command -v hf
    return
  fi
  if [[ -x "${python_dir}/huggingface-cli" ]]; then
    printf '%s\n' "${python_dir}/huggingface-cli"
    return
  fi
  if command_exists huggingface-cli; then
    command -v huggingface-cli
    return
  fi
  print_error "huggingface-cli is unavailable after Python dependency bootstrap."
  exit 1
}

ensure_checkpoints() {
  local checkpoint_root="${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints"
  local huggingface_cli_bin
  local required_paths=(
    "liveportrait_onnx/warping_spade-fix.onnx"
    "liveportrait_onnx/landmark.onnx"
    "liveportrait_onnx/motion_extractor.onnx"
    "liveportrait_onnx/retinaface_det_static.onnx"
    "liveportrait_onnx/face_2dpose_106_static.onnx"
    "liveportrait_onnx/appearance_feature_extractor.onnx"
    "liveportrait_onnx/stitching.onnx"
    "liveportrait_onnx/stitching_eye.onnx"
    "liveportrait_onnx/stitching_lip.onnx"
    "JoyVASA/motion_generator/motion_generator_hubert_chinese.pt"
    "JoyVASA/motion_template/motion_template.pkl"
    "chinese-hubert-base/config.json"
  )
  local relative_path
  local missing_paths=()
  local hf_cli_command_name

  if [[ "${RUNPOD_DOWNLOAD_CHECKPOINTS:-1}" != "1" ]]; then
    print_warning "Checkpoint download disabled by RUNPOD_DOWNLOAD_CHECKPOINTS=${RUNPOD_DOWNLOAD_CHECKPOINTS}"
  else
    for relative_path in "${required_paths[@]}"; do
      if [[ ! -e "${checkpoint_root}/${relative_path}" ]]; then
        missing_paths+=("${relative_path}")
      fi
    done
    if ((${#missing_paths[@]} > 0)); then
      huggingface_cli_bin="$(resolve_huggingface_cli_bin "${PYTHON_BIN}")"
      hf_cli_command_name="$(basename "${huggingface_cli_bin}")"

      print_info "Downloading FasterLivePortrait checkpoints from ${DEFAULT_CHECKPOINT_REPO_ID}"
      "${huggingface_cli_bin}" \
        download "${DEFAULT_CHECKPOINT_REPO_ID}" \
        --local-dir "${checkpoint_root}"

      print_info "Downloading JoyVASA checkpoints from ${DEFAULT_JOYVASA_CHECKPOINT_REPO_ID}"
      "${huggingface_cli_bin}" \
        download "${DEFAULT_JOYVASA_CHECKPOINT_REPO_ID}" \
        --local-dir "${checkpoint_root}/JoyVASA"

      print_info "Downloading chinese-hubert-base checkpoints from ${DEFAULT_CHINESE_HUBERT_REPO_ID}"
      "${huggingface_cli_bin}" \
        download "${DEFAULT_CHINESE_HUBERT_REPO_ID}" \
        --local-dir "${checkpoint_root}/chinese-hubert-base"

      if [[ "${hf_cli_command_name}" == "huggingface-cli" ]]; then
        print_warning "huggingface-cli is deprecated. The bootstrap prefers the newer 'hf' command when available."
      fi
    fi
  fi

  missing_paths=()
  for relative_path in "${required_paths[@]}"; do
    if [[ ! -e "${checkpoint_root}/${relative_path}" ]]; then
      missing_paths+=("${relative_path}")
    fi
  done
  if ((${#missing_paths[@]} > 0)); then
    print_error "Missing checkpoints after bootstrap: ${missing_paths[*]}"
    exit 1
  fi
}

clear_prebuilt_trt_engines_if_requested() {
  local engine_root="${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints/liveportrait_onnx"

  if [[ "${RUNPOD_FORCE_TRT_REBUILD:-1}" != "1" ]]; then
    return
  fi

  if [[ ! -d "${engine_root}" ]]; then
    return
  fi

  print_info "Removing bundled TRT engines so this Pod rebuilds device-compatible plans"
  find "${engine_root}" -maxdepth 1 -type f \( -name '*.trt' -o -name '*.engine_ready.*.txt' \) -delete
}

generate_api_token() {
  local python_bin="$1"
  "${python_bin}" - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
}

resolve_api_token() {
  local python_bin="$1"
  if [[ -n "${ANIMATION_API_TOKEN:-}" ]]; then
    printf '%s\n' "${ANIMATION_API_TOKEN}"
    return
  fi
  if [[ -f "${TOKEN_FILE_PATH}" ]]; then
    tr -d '\r\n' < "${TOKEN_FILE_PATH}"
    return
  fi
  mkdir -p "${STATE_ROOT}"
  generate_api_token "${python_bin}" > "${TOKEN_FILE_PATH}"
  chmod 600 "${TOKEN_FILE_PATH}"
  tr -d '\r\n' < "${TOKEN_FILE_PATH}"
}

ssh_requested() {
  [[ "${RUNPOD_ENABLE_SSH:-0}" == "1" ]] || [[ -n "${RUNPOD_TCP_PORT_22:-}" ]] || [[ -n "${PUBLIC_KEY:-}" ]] || [[ -n "${SSH_PUBLIC_KEY:-}" ]]
}

ensure_sshd() {
  local ssh_public_key_value

  if ! ssh_requested; then
    return
  fi
  if pgrep -x sshd >/dev/null 2>&1; then
    return
  fi

  install_apt_packages_if_missing openssh-server >/dev/null
  mkdir -p /var/run/sshd /root/.ssh
  chmod 700 /root/.ssh
  ssh_public_key_value="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-}}"
  if [[ -n "${ssh_public_key_value}" ]]; then
    printf '%s\n' "${ssh_public_key_value}" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
  else
    print_warning "SSH requested but no PUBLIC_KEY/SSH_PUBLIC_KEY was provided."
  fi
  ssh-keygen -A >/dev/null 2>&1 || true
  /usr/sbin/sshd
  print_info "sshd started"
}

build_proxy_url() {
  local api_port="$1"
  if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
    return
  fi
  printf 'https://%s-%s.proxy.runpod.net/\n' "${RUNPOD_POD_ID}" "${api_port}"
}

build_tcp_url() {
  local api_port="$1"
  local tcp_port_variable_name="RUNPOD_TCP_PORT_${api_port}"
  local mapped_tcp_port="${!tcp_port_variable_name:-}"
  if [[ -z "${RUNPOD_PUBLIC_IP:-}" || -z "${mapped_tcp_port}" ]]; then
    return
  fi
  printf 'http://%s:%s/\n' "${RUNPOD_PUBLIC_IP}" "${mapped_tcp_port}"
}

print_access_summary() {
  local api_port="$1"
  local api_token="$2"
  local proxy_url
  local tcp_url

  proxy_url="$(build_proxy_url "${api_port}")"
  tcp_url="$(build_tcp_url "${api_port}")"

  print_info "Local health URL: http://127.0.0.1:${api_port}/api/health"
  print_info "Token: ${api_token}"
  if [[ -n "${proxy_url}" ]]; then
    print_info "Proxy UI URL: ${proxy_url}?token=${api_token}"
  fi
  if [[ -n "${tcp_url}" ]]; then
    print_info "Direct TCP UI URL: ${tcp_url}?token=${api_token}"
    print_info "Direct TCP is preferred for long-lived websocket streams."
  fi
  if [[ -n "${RUNPOD_PUBLIC_IP:-}" && -n "${RUNPOD_TCP_PORT_22:-}" ]]; then
    print_info "SSH command: ssh root@${RUNPOD_PUBLIC_IP} -p ${RUNPOD_TCP_PORT_22}"
  fi
}

wait_for_healthcheck() {
  local api_port="$1"
  local api_token="$2"
  local healthcheck_url="http://127.0.0.1:${api_port}/api/health"
  local attempt_number

  for ((attempt_number = 1; attempt_number <= HEALTHCHECK_MAX_ATTEMPTS; attempt_number += 1)); do
    if curl -fsS -H "Authorization: Bearer ${api_token}" "${healthcheck_url}" >/dev/null 2>&1; then
      return
    fi
    sleep "${HEALTHCHECK_SLEEP_SECONDS}"
  done

  print_error "Health check failed: ${healthcheck_url}"
  if [[ -f "${LOG_FILE_PATH}" ]]; then
    print_error "Last log lines:"
    tail -n 120 "${LOG_FILE_PATH}" >&2 || true
  fi
  exit 1
}

cleanup() {
  if [[ -n "${API_PID}" ]] && kill -0 "${API_PID}" >/dev/null 2>&1; then
    kill "${API_PID}" >/dev/null 2>&1 || true
    wait "${API_PID}" >/dev/null 2>&1 || true
  fi
}

main() {
  trap cleanup EXIT INT TERM

  ensure_runpod_system_dependencies

  PYTHON_BIN="$(resolve_python_bin)"
  PIP_BIN="$(resolve_pip_bin "${PYTHON_BIN}")"
  ensure_project_layout
  ensure_faster_liveportrait_repo
  apply_faster_liveportrait_overrides_if_present
  repair_nvidia_driver_links_if_needed
  ensure_gpu_runtime "${PYTHON_BIN}"
  ensure_trt_builder_runtime "${PYTHON_BIN}"
  ensure_python_runtime_modules "${PYTHON_BIN}" "${PIP_BIN}"

  export ANIMATION_API_HOST="${ANIMATION_API_HOST:-${DEFAULT_API_HOST}}"
  export ANIMATION_API_PORT="${ANIMATION_API_PORT:-${DEFAULT_API_PORT}}"
  export ANIMATION_BACKEND="${ANIMATION_BACKEND:-${DEFAULT_BACKEND}}"
  export ANIMATION_TRT_RUNTIME="${ANIMATION_TRT_RUNTIME:-${DEFAULT_TRT_RUNTIME}}"
  export ANIMATION_TRT_PRECISION="${ANIMATION_TRT_PRECISION:-${DEFAULT_TRT_PRECISION}}"
  export ANIMATION_WARMUP_ENABLED="${ANIMATION_WARMUP_ENABLED:-1}"
  export ANIMATION_IDLE_VIDEO="${ANIMATION_IDLE_VIDEO:-${DEFAULT_IDLE_VIDEO_PATH}}"
  export ANIMATION_API_TOKEN="$(resolve_api_token "${PYTHON_BIN}")"
  export ANIMATION_VIDEO_ENCODER="${ANIMATION_VIDEO_ENCODER:-cpu}"

  if [[ "${ANIMATION_BACKEND}" != "${DEFAULT_BACKEND}" ]]; then
    print_error "Runpod bootstrap only supports ANIMATION_BACKEND=${DEFAULT_BACKEND}. Current value: ${ANIMATION_BACKEND}"
    exit 1
  fi

  ensure_idle_video_exists
  ensure_checkpoints
  ensure_trt_plugin_runtime "${PYTHON_BIN}"
  ensure_source_assets_exist
  clear_prebuilt_trt_engines_if_requested
  ensure_sshd

  print_info "Starting realtime_stream_api.py with backend=${ANIMATION_BACKEND} trt_runtime=${ANIMATION_TRT_RUNTIME}"
  mkdir -p "$(dirname "${LOG_FILE_PATH}")"
  : > "${LOG_FILE_PATH}"

  (
    cd "${PROJECT_ROOT}"
    set -o pipefail
    "${PYTHON_BIN}" realtime_stream_api.py \
      --host "${ANIMATION_API_HOST}" \
      --port "${ANIMATION_API_PORT}" \
      --backend "${ANIMATION_BACKEND}" \
      --trt-runtime "${ANIMATION_TRT_RUNTIME}" \
      --trt-precision "${ANIMATION_TRT_PRECISION}" \
      > >(tee -a "${LOG_FILE_PATH}") 2>&1
  ) &
  API_PID=$!

  wait_for_healthcheck "${ANIMATION_API_PORT}" "${ANIMATION_API_TOKEN}"
  print_access_summary "${ANIMATION_API_PORT}" "${ANIMATION_API_TOKEN}"

  wait "${API_PID}"
}

main "$@"
