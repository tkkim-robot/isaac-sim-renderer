#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

examples=(
  examples/01_crazyflie_reach_avoid.py
  examples/02_mobile_robot_reach_avoid.py
  examples/03_differential_drive_dynamics.py
)

for example in "${examples[@]}"; do
  echo "Rendering ${example}"
  "${SCRIPT_DIR}/run-example.sh" "${example}" "$@"
done

echo "All tutorials rendered under outputs/."
