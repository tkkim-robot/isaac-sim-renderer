#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/_common.sh
source "${SCRIPT_DIR}/_common.sh"

cd -- "${REPO_ROOT}"
python -m pytest -q
python -m ruff check .
python -m compileall -q controllers isaac_renderer examples

for shell_file in scripts/*.sh; do
  bash -n "${shell_file}"
done

if command -v xmllint >/dev/null 2>&1; then
  xmllint --noout assets/robots/*.urdf
fi

validate_compose
echo "Host tests and static validation passed."
