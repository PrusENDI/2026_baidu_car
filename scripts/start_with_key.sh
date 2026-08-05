#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BACKEND_PATH="${PROJECT_ROOT}/smartcar/paddlebaidu/infer_cs/base/infer_back_end.py"
PROBE_PATH="${SCRIPT_DIR}/inference_backend_probe.py"
KEY_WAITER_PATH="${SCRIPT_DIR}/wait_for_start_key.py"
MAIN_PATH="${PROJECT_ROOT}/car_start_2026.py"
WATCHDOG_INTERVAL="${WATCHDOG_INTERVAL:-2}"
RESTART_DELAY="${RESTART_DELAY:-2}"
LOCK_PATH="${XDG_RUNTIME_DIR:-/tmp}/baidu-smart-key-start.lock"

WATCHDOG_PID=""
MAIN_PID=""

log() {
    printf '%s %s\n' "$(date -Is)" "$*"
}

backend_running() {
    pgrep -f -- "${BACKEND_PATH}" >/dev/null 2>&1
}

start_backend_if_needed() {
    if backend_running; then
        return 0
    fi
    log "inference backend is not running; starting it"
    "${PYTHON_BIN}" "${BACKEND_PATH}" &
}

backend_watchdog() {
    while true; do
        start_backend_if_needed
        sleep "${WATCHDOG_INTERVAL}"
    done
}

cleanup() {
    trap - INT TERM EXIT
    if [[ -n "${MAIN_PID}" ]]; then
        kill "${MAIN_PID}" 2>/dev/null || true
        wait "${MAIN_PID}" 2>/dev/null || true
    fi
    if [[ -n "${WATCHDOG_PID}" ]]; then
        kill "${WATCHDOG_PID}" 2>/dev/null || true
        wait "${WATCHDOG_PID}" 2>/dev/null || true
    fi
}

exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    log "another key-start supervisor is already running"
    exit 1
fi

trap cleanup INT TERM EXIT
cd "${PROJECT_ROOT}"

backend_watchdog &
WATCHDOG_PID=$!

while true; do
    log "waiting for inference backend readiness"
    "${PYTHON_BIN}" "${KEY_WAITER_PATH}" --show-loading || true
    "${PYTHON_BIN}" "${PROBE_PATH}" --wait

    log "waiting for MC602 port 5 key value 12"
    "${PYTHON_BIN}" "${KEY_WAITER_PATH}"
    key_status=$?
    if [[ "${key_status}" -ne 0 ]]; then
        log "key listener exited without a start event: ${key_status}"
        sleep "${RESTART_DELAY}"
        continue
    fi

    log "start event received; rechecking inference backend"
    "${PYTHON_BIN}" "${PROBE_PATH}" --wait

    log "starting full task program"
    "${PYTHON_BIN}" "${MAIN_PATH}" &
    MAIN_PID=$!
    wait "${MAIN_PID}"
    main_status=$?
    MAIN_PID=""
    log "full task program exited with status ${main_status}; returning to key wait"
    sleep "${RESTART_DELAY}"
done
