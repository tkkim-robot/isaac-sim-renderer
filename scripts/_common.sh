#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.yaml"
COMPOSE=(
  docker compose
  --project-directory "${REPO_ROOT}"
  --file "${COMPOSE_FILE}"
)

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Required command not found: ${command_name}" >&2
    return 1
  fi
}

prepare_output_directory() {
  local output_dir="${REPO_ROOT}/outputs"

  if [[ -L "${output_dir}" ]]; then
    echo "Refusing to use a symbolic link as the output directory: ${output_dir}" >&2
    return 1
  fi
  if [[ -e "${output_dir}" && ! -d "${output_dir}" ]]; then
    echo "Output path exists but is not a directory: ${output_dir}" >&2
    return 1
  fi

  mkdir -p -- "${output_dir}"

  # The host user owns this bind mount while Isaac Sim writes as UID 1234.
  # A sticky shared directory gives both sides write access without recursively
  # changing ownership elsewhere in the checkout.
  chmod 1777 -- "${output_dir}"
}

validate_compose() {
  require_command docker
  "${COMPOSE[@]}" config --quiet
}
