#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if (( $# == 0 )); then
  echo "Usage: scripts/run-example.sh <python-script> [arguments ...]" >&2
  echo "Example: scripts/run-example.sh examples/01_crazyflie_reach_avoid.py" >&2
  exit 2
fi

"${SCRIPT_DIR}/prepare.sh"
exec "${COMPOSE[@]}" run --rm --no-deps --build isaac-sim "$@"
