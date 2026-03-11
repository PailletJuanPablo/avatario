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
DEFAULT_TENSORRT_PIP_PACKAGES=(
  "tensorrt-cu12"
  "tensorrt-cu12-bindings"
  "tensorrt-cu12-libs"
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
  print_error "Python runtime not found. Use a TensorRT-capable image such as shaoguo/faster_liveportrait:v3."
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
  print_error "pip runtime not found. Use a TensorRT-capable image such as shaoguo/faster_liveportrait:v3."
  exit 1
}

ensure_runpod_system_dependencies() {
  install_apt_packages_if_missing git curl ffmpeg ca-certificates >/dev/null
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

ensure_tensorrt_module() {
  local python_bin="$1"
  local pip_bin="$2"
  local tensorrt_missing

  tensorrt_missing="$("${python_bin}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("tensorrt") is None else "0")
PY
)"
  if [[ "${tensorrt_missing}" != "1" ]]; then
    return
  fi

  print_info "Installing TensorRT Python packages for CUDA 12"
  "${pip_bin}" install --no-cache-dir "${DEFAULT_TENSORRT_PIP_PACKAGES[@]}"

  tensorrt_missing="$("${python_bin}" - <<'PY'
import importlib.util
print("1" if importlib.util.find_spec("tensorrt") is None else "0")
PY
)"
  if [[ "${tensorrt_missing}" == "1" ]]; then
    print_error "TensorRT Python module is still unavailable after installation."
    exit 1
  fi
}

ensure_python_runtime_modules() {
  local python_bin="$1"
  local pip_bin="$2"
  local missing_core_modules
  local missing_extension_modules

  missing_core_modules="$("${python_bin}" - <<'PY'
import importlib.util

core_modules = ("cv2", "numpy", "torch")
missing_modules = [name for name in core_modules if importlib.util.find_spec(name) is None]
print(" ".join(missing_modules))
PY
)"

  if [[ -n "${missing_core_modules}" ]]; then
    print_error "The selected image is missing required TRT runtime modules: ${missing_core_modules}"
    print_error "Use a Runpod PyTorch image with CUDA support."
    exit 1
  fi

  missing_extension_modules="$("${python_bin}" - <<'PY'
import importlib.util

extension_modules = ("aiortc", "av", "fastapi", "huggingface_hub", "multipart", "omegaconf", "transformers", "uvicorn")
missing_modules = [name for name in extension_modules if importlib.util.find_spec(name) is None]
print(" ".join(missing_modules))
PY
)"

  if [[ -z "${missing_extension_modules}" ]]; then
    return
  fi

  print_info "Installing Python packages for realtime_stream_api.py"
  "${pip_bin}" install --no-cache-dir \
    aiortc==1.14.0 \
    av \
    fastapi \
    "huggingface_hub[cli]" \
    omegaconf \
    python-multipart \
    transformers==4.40.2 \
    "uvicorn[standard]"
}

ensure_project_layout() {
  mkdir -p \
    "${PROJECT_ROOT}/inputs" \
    "${PROJECT_ROOT}/output" \
    "${PROJECT_ROOT}/output_fasterliveportrait" \
    "${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints" \
    "${PROJECT_ROOT}/third_party/FasterLivePortrait/results" \
    "${STATE_ROOT}"
}

ensure_faster_liveportrait_repo() {
  if [[ -f "${PROJECT_ROOT}/third_party/FasterLivePortrait/run.py" ]]; then
    return
  fi
  print_info "Bootstrapping FasterLivePortrait dependency"
  bash "${PROJECT_ROOT}/scripts/bootstrap_faster_liveportrait.sh"
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

resolve_huggingface_cli_bin() {
  local python_bin="$1"
  local python_dir
  python_dir="$(cd "$(dirname "${python_bin}")" && pwd)"
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
      print_info "Downloading checkpoints from ${DEFAULT_CHECKPOINT_REPO_ID}"
      "${huggingface_cli_bin}" \
        download "${DEFAULT_CHECKPOINT_REPO_ID}" \
        --local-dir "${checkpoint_root}"
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
  ensure_gpu_runtime "${PYTHON_BIN}"
  ensure_tensorrt_module "${PYTHON_BIN}" "${PIP_BIN}"
  ensure_python_runtime_modules "${PYTHON_BIN}" "${PIP_BIN}"
  ensure_project_layout
  ensure_faster_liveportrait_repo

  export ANIMATION_API_HOST="${ANIMATION_API_HOST:-${DEFAULT_API_HOST}}"
  export ANIMATION_API_PORT="${ANIMATION_API_PORT:-${DEFAULT_API_PORT}}"
  export ANIMATION_BACKEND="${ANIMATION_BACKEND:-${DEFAULT_BACKEND}}"
  export ANIMATION_TRT_RUNTIME="${ANIMATION_TRT_RUNTIME:-${DEFAULT_TRT_RUNTIME}}"
  export ANIMATION_TRT_PRECISION="${ANIMATION_TRT_PRECISION:-${DEFAULT_TRT_PRECISION}}"
  export ANIMATION_WARMUP_ENABLED="${ANIMATION_WARMUP_ENABLED:-1}"
  export ANIMATION_IDLE_VIDEO="${ANIMATION_IDLE_VIDEO:-${DEFAULT_IDLE_VIDEO_PATH}}"
  export ANIMATION_API_TOKEN="$(resolve_api_token "${PYTHON_BIN}")"

  ensure_idle_video_exists
  ensure_checkpoints
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
