#!/usr/bin/env bash

set -euo pipefail

readonly DEFAULT_GRADIO_HOST="0.0.0.0"
readonly DEFAULT_GRADIO_PORT="7860"
readonly DEFAULT_WORKSPACE_ROOT="/workspace/musetalk"
readonly DEFAULT_MODELS_DIR_NAME="models"
readonly DEFAULT_RESULTS_DIR_NAME="results"
readonly DEFAULT_MODELS_READY_FILE_NAME=".weights_ready"

print_info() {
  printf '[info] %s\n' "$1"
}

print_warning() {
  printf '[warn] %s\n' "$1" >&2
}

read_env_or_default() {
  local env_key="$1"
  local default_value="$2"
  local raw_value="${!env_key:-}"
  if [[ -z "${raw_value}" ]]; then
    printf '%s\n' "${default_value}"
    return
  fi
  printf '%s\n' "${raw_value}"
}

ensure_directory() {
  local directory_path="$1"
  mkdir -p "${directory_path}"
}

ensure_symlink_target() {
  local source_path="$1"
  local target_path="$2"
  ensure_directory "$(dirname "${target_path}")"
  if [[ -L "${target_path}" ]]; then
    rm -f "${target_path}"
  elif [[ -d "${target_path}" ]]; then
    rm -rf "${target_path}"
  elif [[ -f "${target_path}" ]]; then
    rm -f "${target_path}"
  fi
  ln -s "${source_path}" "${target_path}"
}

ensure_workspace_links() {
  local repo_dir="$1"
  local workspace_root="$2"
  local models_dir="${workspace_root}/${DEFAULT_MODELS_DIR_NAME}"
  local results_dir="${workspace_root}/${DEFAULT_RESULTS_DIR_NAME}"

  ensure_directory "${models_dir}"
  ensure_directory "${results_dir}"

  ensure_symlink_target "${models_dir}" "${repo_dir}/models"
  ensure_symlink_target "${results_dir}" "${repo_dir}/results"
}

download_model_weights_if_needed() {
  local repo_dir="$1"
  local workspace_root="$2"
  local ready_file_path="${workspace_root}/${DEFAULT_MODELS_DIR_NAME}/${DEFAULT_MODELS_READY_FILE_NAME}"
  local skip_download="${MUSETALK_SKIP_WEIGHTS_DOWNLOAD:-0}"

  if [[ "${skip_download}" == "1" ]]; then
    print_warning "Skipping MuseTalk weight download because MUSETALK_SKIP_WEIGHTS_DOWNLOAD=1"
    return
  fi

  if [[ -f "${ready_file_path}" ]]; then
    print_info "MuseTalk weights already available at ${ready_file_path}"
    return
  fi

  print_info "Downloading MuseTalk weights via official download_weights.sh"
  (
    cd "${repo_dir}"
    bash download_weights.sh
  )

  ensure_directory "$(dirname "${ready_file_path}")"
  touch "${ready_file_path}"
}

launch_gradio_app() {
  local repo_dir="$1"
  local gradio_host="$2"
  local gradio_port="$3"

  export GRADIO_SERVER_NAME="${gradio_host}"
  export GRADIO_SERVER_PORT="${gradio_port}"
  export PYTHONPATH="${repo_dir}:${PYTHONPATH:-}"

  print_info "Launching MuseTalk Gradio UI on ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"
  cd "${repo_dir}"
  exec python app.py
}

main() {
  local repo_dir
  local workspace_root
  local gradio_host
  local gradio_port

  repo_dir="$(read_env_or_default "MUSETALK_REPO_DIR" "/opt/MuseTalk")"
  workspace_root="$(read_env_or_default "MUSETALK_WORKSPACE_ROOT" "${DEFAULT_WORKSPACE_ROOT}")"
  gradio_host="$(read_env_or_default "MUSETALK_GRADIO_HOST" "${DEFAULT_GRADIO_HOST}")"
  gradio_port="$(read_env_or_default "MUSETALK_GRADIO_PORT" "${DEFAULT_GRADIO_PORT}")"

  ensure_workspace_links "${repo_dir}" "${workspace_root}"
  download_model_weights_if_needed "${repo_dir}" "${workspace_root}"
  launch_gradio_app "${repo_dir}" "${gradio_host}" "${gradio_port}"
}

main "$@"
