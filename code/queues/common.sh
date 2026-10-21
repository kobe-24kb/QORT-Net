#!/usr/bin/env bash
set -euo pipefail
PY="${PY:-python}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run() {
    echo "=== $(date '+%F %T') START $*"
    local status=0
    "$PY" "$@" || status=$?
    echo "=== $(date '+%F %T') EXIT $status $*"
    return "$status"
}
