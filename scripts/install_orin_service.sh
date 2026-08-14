#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="baidu-smart.service"
EXPECTED_ROOT="/home/jetson/workspaces/baidu_smart_2026_7_17"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SERVICE_SOURCE="${PROJECT_ROOT}/systemd/${SERVICE_NAME}"
SERVICE_TARGET="/etc/systemd/system/${SERVICE_NAME}"

if [[ "${EUID}" -ne 0 ]]; then
    printf 'Please run this installer with sudo:\n  sudo %q\n' "$0" >&2
    exit 1
fi

if [[ "${PROJECT_ROOT}" != "${EXPECTED_ROOT}" ]]; then
    printf 'Project path mismatch. Expected %s, got %s\n' \
        "${EXPECTED_ROOT}" "${PROJECT_ROOT}" >&2
    exit 1
fi

if ! id jetson >/dev/null 2>&1; then
    printf 'Required service user "jetson" does not exist.\n' >&2
    exit 1
fi

install -m 0644 "${SERVICE_SOURCE}" "${SERVICE_TARGET}"
systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

printf '\nInstalled and started %s.\n' "${SERVICE_NAME}"
systemctl --no-pager --full status "${SERVICE_NAME}" || true
printf '\nFollow startup logs with:\n  journalctl -u %s -f\n' "${SERVICE_NAME}"
