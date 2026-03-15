#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_PYTHON_BIN="/root/miniconda3/bin/python"
DEFAULT_API_PORT="8010"
DEFAULT_PLUGIN_PATH="${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so"
TOKEN_FILE_PATH="${PROJECT_ROOT}/.runpod/api_token"
LOG_FILE_PATH="${PROJECT_ROOT}/output_fasterliveportrait/runpod_api.log"
DEFAULT_WARMUP_MAX_ATTEMPTS="${RUNPOD_VALIDATE_WARMUP_MAX_ATTEMPTS:-360}"
DEFAULT_WARMUP_SLEEP_SECONDS="${RUNPOD_VALIDATE_WARMUP_SLEEP_SECONDS:-5}"

print_info() {
  printf '[info] %s\n' "$1"
}

print_error() {
  printf '[error] %s\n' "$1" >&2
}

resolve_python_bin() {
  if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
    printf '%s\n' "${DEFAULT_PYTHON_BIN}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  print_error "Python runtime not found."
  exit 1
}

resolve_api_token() {
  if [[ -n "${ANIMATION_API_TOKEN:-}" ]]; then
    printf '%s\n' "${ANIMATION_API_TOKEN}"
    return
  fi
  if [[ -f "${TOKEN_FILE_PATH}" ]]; then
    tr -d '\r\n' < "${TOKEN_FILE_PATH}"
    return
  fi
  print_error "API token not found. Export ANIMATION_API_TOKEN or ensure ${TOKEN_FILE_PATH} exists."
  exit 1
}

read_health_payload() {
  local api_port="$1"
  local api_token="$2"

  curl -fsS -H "Authorization: Bearer ${api_token}" "http://127.0.0.1:${api_port}/api/health"
}

wait_for_warmup_completion() {
  local python_bin="$1"
  local api_port="$2"
  local api_token="$3"
  local attempt_number
  local payload
  local phase=""
  local progress=""
  local error_message=""
  local warmup_enabled=""
  local warmup_running=""

  for ((attempt_number = 1; attempt_number <= DEFAULT_WARMUP_MAX_ATTEMPTS; attempt_number += 1)); do
    payload="$(read_health_payload "${api_port}" "${api_token}")"
    mapfile -t warmup_fields < <(
      "${python_bin}" -c 'import json, sys; payload = json.loads(sys.argv[1]); print("true" if payload.get("warmupEnabled") else "false"); print("true" if payload.get("warmupRunning") else "false"); print(payload.get("warmupPhase", "")); print(payload.get("warmupProgress", "")); print(payload.get("warmupError", ""))' "${payload}"
    )

    warmup_enabled="${warmup_fields[0]:-false}"
    warmup_running="${warmup_fields[1]:-false}"
    phase="${warmup_fields[2]:-}"
    progress="${warmup_fields[3]:-}"
    error_message="${warmup_fields[4]:-}"

    if [[ -n "${error_message}" ]]; then
      print_error "Warmup failed: ${error_message}"
      printf '%s\n' "${payload}"
      exit 1
    fi
    if [[ "${warmup_enabled}" != "true" ]]; then
      print_info "Warmup is disabled; skipping warmup wait."
      return
    fi
    if [[ "${warmup_running}" != "true" && "${phase}" == "completed" ]]; then
      print_info "Warmup completed successfully."
      printf '%s\n' "${payload}"
      return
    fi

    print_info "Warmup in progress: phase=${phase:-unknown} progress=${progress:-unknown}"
    sleep "${DEFAULT_WARMUP_SLEEP_SECONDS}"
  done

  print_error "Warmup did not complete within the validation timeout."
  exit 1
}

main() {
  local python_bin
  local api_port="${ANIMATION_API_PORT:-${DEFAULT_API_PORT}}"
  local api_token

  python_bin="$(resolve_python_bin)"
  api_token="$(resolve_api_token)"

  print_info "Validating CUDA, TensorRT builder, and TRT plugin"
  "${python_bin}" - <<PY
import ctypes
import os

import pycuda.autoinit  # noqa: F401
import tensorrt as trt
import torch

plugin_path = os.path.abspath(${DEFAULT_PLUGIN_PATH@Q})

print("torch_cuda_available =", torch.cuda.is_available())
print("torch_device_count =", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch_device_0 =", torch.cuda.get_device_name(0))

builder = trt.Builder(trt.Logger(trt.Logger.ERROR))
print("trt_builder_ready =", builder is not None)
ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
print("trt_plugin_ready = True")
PY

  print_info "Checking API health on port ${api_port}"
  read_health_payload "${api_port}" "${api_token}"
  printf '\n'
  wait_for_warmup_completion "${python_bin}" "${api_port}" "${api_token}"

  if [[ -f "${LOG_FILE_PATH}" ]]; then
    print_info "Last API log lines"
    tail -n 40 "${LOG_FILE_PATH}"
  fi
}

main "$@"
