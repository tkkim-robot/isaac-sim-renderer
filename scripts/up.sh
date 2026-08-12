#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

"${SCRIPT_DIR}/prepare.sh"
exec "${COMPOSE[@]}" up --build --detach "$@"
