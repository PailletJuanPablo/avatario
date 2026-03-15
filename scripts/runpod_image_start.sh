#!/usr/bin/env bash

set -euo pipefail

DEFAULT_IMAGE_APP_ROOT="/app"

print_info() {
  printf '[info] %s\n' "$1"
}

main() {
  local image_app_root="${RUNPOD_IMAGE_APP_ROOT:-${DEFAULT_IMAGE_APP_ROOT}}"
  local entrypoint_script_path="${image_app_root}/scripts/runpod_entrypoint.sh"

  if [[ ! -d "${image_app_root}" ]]; then
    printf '[error] Image app root not found: %s\n' "${image_app_root}" >&2
    exit 1
  fi
  if [[ ! -f "${entrypoint_script_path}" ]]; then
    printf '[error] Runpod entrypoint script not found: %s\n' "${entrypoint_script_path}" >&2
    exit 1
  fi

  print_info "Cloning repository into workspace and bootstrapping from there"
  cd "${image_app_root}"
  exec bash "${entrypoint_script_path}"
}

main "$@"
