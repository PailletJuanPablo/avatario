#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPO_URL="https://github.com/PailletJuanPablo/avatario.git"
DEFAULT_REPO_REF="main"
DEFAULT_WORKSPACE_ROOT="/workspace"
DEFAULT_REPO_DIR="${DEFAULT_WORKSPACE_ROOT}/animation"

print_info() {
  printf '[info] %s\n' "$1"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

ensure_base_tools() {
  if command_exists git && command_exists curl; then
    return
  fi
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends git curl ca-certificates ffmpeg
}

clone_or_refresh_repo() {
  local repo_url="$1"
  local repo_ref="$2"
  local repo_dir="$3"

  mkdir -p "$(dirname "${repo_dir}")"
  if [[ ! -d "${repo_dir}/.git" ]]; then
    git clone --branch "${repo_ref}" --depth 1 "${repo_url}" "${repo_dir}"
    return
  fi

  if [[ "${RUNPOD_GIT_AUTO_UPDATE:-1}" != "1" ]]; then
    return
  fi

  git -C "${repo_dir}" fetch --depth 1 origin "${repo_ref}"
  git -C "${repo_dir}" checkout "${repo_ref}"
  git -C "${repo_dir}" pull --ff-only origin "${repo_ref}"
}

main() {
  local repo_url="${RUNPOD_GIT_REPO:-${DEFAULT_REPO_URL}}"
  local repo_ref="${RUNPOD_GIT_REF:-${DEFAULT_REPO_REF}}"
  local repo_dir="${RUNPOD_REPO_DIR:-${DEFAULT_REPO_DIR}}"

  ensure_base_tools
  clone_or_refresh_repo "${repo_url}" "${repo_ref}" "${repo_dir}"

  print_info "Launching Runpod PyTorch TRT bootstrap from ${repo_dir}"
  exec bash "${repo_dir}/scripts/runpod_bootstrap.sh"
}

main "$@"
