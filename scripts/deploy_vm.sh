#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BOOTSTRAP_SCRIPT_PATH="${PROJECT_ROOT}/scripts/bootstrap_faster_liveportrait.sh"
ENV_FILE_PATH="${PROJECT_ROOT}/.env"
DEFAULT_ENV_FILE_PATH="${PROJECT_ROOT}/.env.example"
DOCKER_COMPOSE_FILE_PATH="${PROJECT_ROOT}/docker-compose.yml"
DEFAULT_IDLE_VIDEO_PATH="inputs/idlevid.mp4"
HEALTH_ENDPOINT_PATH="/api/health"
HEALTH_CHECK_MAX_ATTEMPTS=60
HEALTH_CHECK_SLEEP_SECONDS=2
DOCKER_SERVICE_NAME="animation-api"

print_info() {
  printf '[info] %s\n' "$1"
}

print_warning() {
  printf '[warn] %s\n' "$1" >&2
}

print_error() {
  printf '[error] %s\n' "$1" >&2
}

require_command() {
  local command_name="$1"
  if command -v "${command_name}" >/dev/null 2>&1; then
    return
  fi
  print_error "Required command not found: ${command_name}"
  exit 1
}

ensure_file_exists() {
  local path_value="$1"
  if [[ -f "${path_value}" ]]; then
    return
  fi
  print_error "Required file not found: ${path_value}"
  exit 1
}

ensure_directory_exists() {
  local path_value="$1"
  mkdir -p "${path_value}"
}

ensure_faster_liveportrait_repo() {
  ensure_file_exists "${BOOTSTRAP_SCRIPT_PATH}"
  bash "${BOOTSTRAP_SCRIPT_PATH}"
  ensure_file_exists "${PROJECT_ROOT}/third_party/FasterLivePortrait/run.py"
}

ensure_env_file() {
  if [[ -f "${ENV_FILE_PATH}" ]]; then
    return
  fi
  ensure_file_exists "${DEFAULT_ENV_FILE_PATH}"
  cp "${DEFAULT_ENV_FILE_PATH}" "${ENV_FILE_PATH}"
  print_info "Created .env from .env.example"
}

load_env_file() {
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE_PATH}"
  set +a
}

ensure_runtime_directories() {
  ensure_directory_exists "${PROJECT_ROOT}/inputs"
  ensure_directory_exists "${PROJECT_ROOT}/output"
  ensure_directory_exists "${PROJECT_ROOT}/output_fasterliveportrait"
  ensure_directory_exists "${PROJECT_ROOT}/third_party/FasterLivePortrait/checkpoints"
  ensure_directory_exists "${PROJECT_ROOT}/third_party/FasterLivePortrait/results"
}

ensure_idle_video() {
  local configured_idle_video="${ANIMATION_IDLE_VIDEO:-${DEFAULT_IDLE_VIDEO_PATH}}"
  local configured_idle_video_path="${PROJECT_ROOT}/${configured_idle_video}"

  if [[ -f "${configured_idle_video_path}" ]]; then
    return
  fi

  print_error "Idle video not found: ${configured_idle_video_path}"
  print_error "Set ANIMATION_IDLE_VIDEO in .env to an existing file under ${PROJECT_ROOT}/inputs."
  exit 1
}

run_compose_up() {
  (
    cd "${PROJECT_ROOT}"
    docker compose -f "${DOCKER_COMPOSE_FILE_PATH}" up --build -d
  )
}

build_health_check_url() {
  local api_port="${ANIMATION_API_PORT:-8010}"
  printf 'http://127.0.0.1:%s%s' "${api_port}" "${HEALTH_ENDPOINT_PATH}"
}

wait_for_healthcheck() {
  local healthcheck_url
  local curl_arguments
  local token_value
  local attempt_number

  healthcheck_url="$(build_health_check_url)"
  token_value="${ANIMATION_API_TOKEN:-}"
  curl_arguments=(-fsS "${healthcheck_url}")
  if [[ -n "${token_value}" ]]; then
    curl_arguments=(-fsS -H "Authorization: Bearer ${token_value}" "${healthcheck_url}")
  fi

  for ((attempt_number = 1; attempt_number <= HEALTH_CHECK_MAX_ATTEMPTS; attempt_number += 1)); do
    if curl "${curl_arguments[@]}" >/dev/null 2>&1; then
      print_info "Health check passed: ${healthcheck_url}"
      return
    fi
    sleep "${HEALTH_CHECK_SLEEP_SECONDS}"
  done

  print_warning "Health check did not pass after ${HEALTH_CHECK_MAX_ATTEMPTS} attempts."
  (
    cd "${PROJECT_ROOT}"
    docker compose -f "${DOCKER_COMPOSE_FILE_PATH}" logs --tail 120 "${DOCKER_SERVICE_NAME}" || true
  )
  exit 1
}

main() {
  require_command docker
  require_command curl
  require_command git
  ensure_file_exists "${DOCKER_COMPOSE_FILE_PATH}"
  ensure_faster_liveportrait_repo
  ensure_env_file
  load_env_file
  ensure_runtime_directories
  ensure_idle_video
  run_compose_up
  wait_for_healthcheck
}

main "$@"
