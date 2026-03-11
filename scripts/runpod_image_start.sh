#!/usr/bin/env bash

set -euo pipefail

DEFAULT_IMAGE_APP_ROOT="/app"
DEFAULT_WORKSPACE_ROOT="/workspace"
DEFAULT_WORKSPACE_REPO_DIR="${DEFAULT_WORKSPACE_ROOT}/animation"
DEFAULT_RUNPOD_START_SCRIPT="/start.sh"

print_info() {
  printf '[info] %s\n' "$1"
}

start_runpod_base_services() {
  local start_script_path="${RUNPOD_BASE_START_SCRIPT:-${DEFAULT_RUNPOD_START_SCRIPT}}"
  if [[ ! -x "${start_script_path}" ]]; then
    printf '[warn] Runpod base start script not found or not executable: %s\n' "${start_script_path}" >&2
    return
  fi

  print_info "Starting Runpod base services from ${start_script_path}"
  "${start_script_path}" >/tmp/runpod-base-start.log 2>&1 &
}

main() {
  local image_app_root="${RUNPOD_IMAGE_APP_ROOT:-${DEFAULT_IMAGE_APP_ROOT}}"
  local workspace_repo_dir="${RUNPOD_REPO_DIR:-${DEFAULT_WORKSPACE_REPO_DIR}}"

  if [[ ! -d "${image_app_root}" ]]; then
    printf '[error] Image app root not found: %s\n' "${image_app_root}" >&2
    exit 1
  fi

  start_runpod_base_services

  mkdir -p "${workspace_repo_dir}"
  print_info "Syncing image contents into ${workspace_repo_dir}"
  cp -a "${image_app_root}/." "${workspace_repo_dir}/"

  cd "${workspace_repo_dir}"
  exec bash "${workspace_repo_dir}/scripts/runpod_bootstrap.sh"
}

main "$@"
