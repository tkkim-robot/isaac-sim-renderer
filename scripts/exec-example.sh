#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if (( $# == 0 )); then
  echo "Usage: scripts/exec-example.sh <python-script> [arguments ...]" >&2
  echo "Example: scripts/exec-example.sh examples/01_crazyflie_reach_avoid.py" >&2
  exit 2
fi

prepare_output_directory
validate_compose
exec "${COMPOSE[@]}" exec isaac-sim /isaac-sim/python.sh "$@"
