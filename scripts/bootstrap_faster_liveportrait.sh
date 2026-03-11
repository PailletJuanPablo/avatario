#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_ROOT="${PROJECT_ROOT}/third_party"
FASTER_LIVEPORTRAIT_DIR="${THIRD_PARTY_ROOT}/FasterLivePortrait"
FASTER_LIVEPORTRAIT_REPO_URL="https://github.com/warmshao/FasterLivePortrait.git"
FASTER_LIVEPORTRAIT_ENTRYPOINT="run.py"

print_info() {
  printf '[info] %s\n' "$1"
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

clone_faster_liveportrait_repo() {
  mkdir -p "${THIRD_PARTY_ROOT}"
  git clone --depth 1 "${FASTER_LIVEPORTRAIT_REPO_URL}" "${FASTER_LIVEPORTRAIT_DIR}"
  print_info "Cloned FasterLivePortrait into ${FASTER_LIVEPORTRAIT_DIR}"
}

main() {
  require_command git

  if [[ -f "${FASTER_LIVEPORTRAIT_DIR}/${FASTER_LIVEPORTRAIT_ENTRYPOINT}" ]]; then
    print_info "FasterLivePortrait already present at ${FASTER_LIVEPORTRAIT_DIR}"
    return
  fi

  if [[ -e "${FASTER_LIVEPORTRAIT_DIR}" ]]; then
    print_error "Path exists but FasterLivePortrait is incomplete: ${FASTER_LIVEPORTRAIT_DIR}"
    exit 1
  fi

  clone_faster_liveportrait_repo
}

main "$@"
