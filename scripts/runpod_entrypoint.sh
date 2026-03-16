#!/usr/bin/env bash

set -euo pipefail

DEFAULT_REPO_URL="https://github.com/PailletJuanPablo/avatario.git"
DEFAULT_GIT_REF="main"
DEFAULT_WORKSPACE_ROOT="/workspace"
DEFAULT_REPO_DIR="${DEFAULT_WORKSPACE_ROOT}/animation"
DEFAULT_IMAGE_BUNDLE_DIR="/app/runpod-bundle"

print_info() {
  printf '[info] %s\n' "$1"
}

print_warning() {
  printf '[warn] %s\n' "$1" >&2
}

print_error() {
  printf '[error] %s\n' "$1" >&2
}

print_repo_revision() {
  local repo_dir="$1"
  local git_ref

  if [[ ! -d "${repo_dir}/.git" ]]; then
    return
  fi

  git_ref="$(git -C "${repo_dir}" rev-parse --short HEAD 2>/dev/null || true)"
  if [[ -n "${git_ref}" ]]; then
    print_info "Repository revision: ${git_ref}"
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

install_entrypoint_dependencies() {
  local packages_to_install=()
  if command_exists git && command_exists curl && command_exists update-ca-certificates; then
    return
  fi
  if ! command_exists apt-get; then
    print_error "apt-get is unavailable; install git and curl in the base image first."
    exit 1
  fi
  if ! command_exists git; then
    packages_to_install+=("git")
  fi
  if ! command_exists curl; then
    packages_to_install+=("curl")
  fi
  if ! command_exists update-ca-certificates; then
    packages_to_install+=("ca-certificates")
  fi
  if ((${#packages_to_install[@]} == 0)); then
    return
  fi
  print_info "Installing entrypoint packages: ${packages_to_install[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends "${packages_to_install[@]}"
}

clone_or_update_repo() {
  local repo_url="$1"
  local git_ref="$2"
  local repo_dir="$3"

  mkdir -p "$(dirname "${repo_dir}")"

  if [[ ! -d "${repo_dir}/.git" ]]; then
    print_info "Cloning repository ${repo_url} (${git_ref}) into ${repo_dir}"
    git clone --branch "${git_ref}" --depth 1 "${repo_url}" "${repo_dir}"
    print_repo_revision "${repo_dir}"
    return
  fi

  print_info "Repository already present at ${repo_dir}"
  if [[ "${RUNPOD_GIT_AUTO_UPDATE:-0}" != "1" ]]; then
    print_warning "RUNPOD_GIT_AUTO_UPDATE=0; keeping the existing checkout on the volume"
    print_repo_revision "${repo_dir}"
    return
  fi

  print_info "Updating repository to ${git_ref}"
  if ! git -C "${repo_dir}" fetch --depth 1 origin "${git_ref}"; then
    print_warning "git fetch failed; keeping existing checkout"
    return
  fi
  if ! git -C "${repo_dir}" checkout "${git_ref}"; then
    print_warning "git checkout failed; keeping existing checkout"
    return
  fi
  if ! git -C "${repo_dir}" pull --ff-only origin "${git_ref}"; then
    print_warning "git pull failed; keeping existing checkout"
    print_repo_revision "${repo_dir}"
    return
  fi
  print_repo_revision "${repo_dir}"
}

apply_image_bundle_overrides() {
  local repo_dir="$1"
  local bundle_dir="${RUNPOD_IMAGE_BUNDLE_DIR:-${DEFAULT_IMAGE_BUNDLE_DIR}}"
  local relative_path
  local source_path
  local target_path
  local override_paths=(
    "requirements-runpod.txt"
    "scripts/bootstrap_faster_liveportrait.sh"
    "scripts/runpod_bootstrap.sh"
    "scripts/runpod_validate_runtime.sh"
    "faster_liveportrait_runner.py"
  )

  if [[ ! -d "${bundle_dir}" ]]; then
    return
  fi

  for relative_path in "${override_paths[@]}"; do
    source_path="${bundle_dir}/${relative_path}"
    target_path="${repo_dir}/${relative_path}"
    if [[ ! -f "${source_path}" ]]; then
      continue
    fi
    mkdir -p "$(dirname "${target_path}")"
    cp "${source_path}" "${target_path}"
  done
}

main() {
  local repo_url="${RUNPOD_GIT_REPO:-${DEFAULT_REPO_URL}}"
  local git_ref="${RUNPOD_GIT_REF:-${DEFAULT_GIT_REF}}"
  local repo_dir="${RUNPOD_REPO_DIR:-${DEFAULT_REPO_DIR}}"

  install_entrypoint_dependencies
  clone_or_update_repo "${repo_url}" "${git_ref}" "${repo_dir}"
  apply_image_bundle_overrides "${repo_dir}"

  chmod +x "${repo_dir}/scripts/"*.sh >/dev/null 2>&1 || true

  cd "${repo_dir}"
  exec bash "${repo_dir}/scripts/runpod_bootstrap.sh"
}

main "$@"
