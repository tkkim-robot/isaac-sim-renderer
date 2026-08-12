#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

prepare_output_directory
validate_compose

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed, but its daemon is not available to this user." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi is unavailable; Isaac Sim requires an NVIDIA GPU." >&2
else
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

echo "Container prerequisites and ${REPO_ROOT}/outputs are ready."
