#!/usr/bin/env bash
#
# Container entrypoint: validate the configuration, translate environment variables into
# `python -m pyzm.serve` flags, then hand the process over.
#
# This is deliberately the only code of our own that runs at start-up. It is NOT
# orchestration and it adds NO HTTP behaviour -- Constitution Principle III forbids both.
# It does exactly two things upstream cannot do for itself: refuse to start in a
# configuration the operator did not actually ask for (see the preflights below), and turn
# a documented environment table into a command line.
#
# The variable table is a convenience, never a cage: anything passed after the image name
# on `docker run` is appended verbatim to the server command line (RC-4).

set -euo pipefail

# EX_CONFIG from sysexits.h. A configuration error, deliberately distinguishable from a
# crash (1) or a clean exit (0) -- see contracts/container-interface.md section 6.
readonly EX_CONFIG=78

log()  { printf '%s [entrypoint] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }
warn() { log "WARNING: $*" >&2; }
die()  { log "ERROR: $*" >&2; exit "${EX_CONFIG}"; }

# -----------------------------------------------------------------------------
# Preflight 1: GPU (FR-004, RC-2, research R6/R7)
# -----------------------------------------------------------------------------
# Why this exists at all: upstream CANNOT detect a CUDA-less OpenCV, and CANNOT detect a
# CUDA-capable OpenCV with no visible device. `YoloBase._setup_gpu()` checks the OpenCV
# *version*, not its capabilities, and `setPreferableBackend(DNN_BACKEND_CUDA)` succeeds
# even when the backend will silently fall back to the CPU at forward() time. Nothing is
# logged. So the obvious check -- grep the container log for "fell back" -- passes in
# exactly the failure case it exists to catch.
#
# Without this preflight, FR-004's "explicit, visible failure" is unsatisfiable. A
# container that exits saying "you forgot --gpus all" is strictly better than one that runs
# at a fifth of the speed and says nothing, which is how the predecessor stack spent a year.
#
# PYZM_SERVE_ALLOW_CPU is the escape hatch, and it matters for Principle V: someone without
# a GPU may still legitimately want to try this image, and refusing outright with no
# override would be an assumption about their hardware.
preflight_gpu() {
    local processor="${PYZM_SERVE_PROCESSOR:-gpu}"
    local allow_cpu="${PYZM_SERVE_ALLOW_CPU:-}"

    if [[ "${processor}" != "gpu" ]]; then
        if [[ -z "${allow_cpu}" ]]; then
            warn "PYZM_SERVE_PROCESSOR=${processor} -- this is a GPU image and it is about"
            warn "to run inference WITHOUT the GPU. Set PYZM_SERVE_ALLOW_CPU=1 to"
            warn "acknowledge this and silence the warning."
        else
            warn "Running with PYZM_SERVE_PROCESSOR=${processor}: inference will NOT use"
            warn "the GPU. Expect roughly an order of magnitude more latency per frame."
        fi
        return 0
    fi

    local device_count
    device_count="$(python3 -c \
        'import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())' 2>/dev/null || echo error)"

    if [[ "${device_count}" =~ ^[0-9]+$ ]] && (( device_count > 0 )); then
        log "GPU preflight OK: ${device_count} CUDA device(s) visible."
        return 0
    fi

    local detail
    if [[ "${device_count}" == "error" ]]; then
        detail="cv2.cuda.getCudaEnabledDeviceCount() could not be called at all, which
means the OpenCV in this image was not built with CUDA. That should have been impossible:
the build asserts it (scripts/assert-cuda-build.py). Please report this as a bug."
    else
        detail="cv2.cuda.getCudaEnabledDeviceCount() returned ${device_count}."
    fi

    if [[ -n "${allow_cpu}" ]]; then
        warn "================================================================"
        warn "GPU INFERENCE WAS REQUESTED BUT NO CUDA DEVICE IS VISIBLE."
        warn "${detail}"
        warn ""
        warn "PYZM_SERVE_ALLOW_CPU is set, so starting anyway -- but every frame will"
        warn "be processed on the CPU, roughly an order of magnitude slower. This is"
        warn "not a working GPU deployment."
        warn "================================================================"
        return 0
    fi

    die "GPU inference was requested but no CUDA device is visible.

${detail}

Did you pass --gpus all?

  docker run --gpus all -p 5000:5000 <image>

Compose users need a device reservation instead -- see docker-compose.yml. This also
needs the NVIDIA Container Toolkit installed on the host and an NVIDIA driver of the 550
branch or newer.

Refusing to start and quietly serve CPU inference: that failure is invisible from the
outside, and it is the single thing this image exists to prevent.

To run on the CPU deliberately, set PYZM_SERVE_ALLOW_CPU=1."
}

# -----------------------------------------------------------------------------
# Preflight 2: authentication (FR-007, RC-3)
# -----------------------------------------------------------------------------
# Both halves refuse a credential that came from somewhere other than this operator.
#
# --token-secret defaults, upstream, to the literal string `change-me`. That is safe only
# while auth is off. Signing JWTs with a published default means anyone who has read
# upstream's source can mint a valid token.
#
# --auth-password defaults to the empty string upstream. contracts/container-interface.md
# section 5 marks it required-when-auth-on and gives it no default here, so auth-on without
# a password would otherwise start with whatever upstream falls back to.
#
# An authenticated endpoint whose credentials live in someone else's source tree is worse
# than an unauthenticated one, because the operator believes it is protected.
preflight_auth() {
    [[ -n "${PYZM_SERVE_AUTH:-}" ]] || return 0

    if [[ -z "${PYZM_SERVE_TOKEN_SECRET:-}" ]]; then
        die "PYZM_SERVE_AUTH is enabled but PYZM_SERVE_TOKEN_SECRET is not set.

Upstream's --token-secret defaults to the literal string 'change-me', which is published
in its source. Signing tokens with it would let anyone who has read that source mint a
valid token for this gateway, so this image refuses to do it.

Set PYZM_SERVE_TOKEN_SECRET to a long random value, for example:

  PYZM_SERVE_TOKEN_SECRET=\$(openssl rand -hex 32)"
    fi

    if [[ -z "${PYZM_SERVE_AUTH_PASSWORD:-}" ]]; then
        die "PYZM_SERVE_AUTH is enabled but PYZM_SERVE_AUTH_PASSWORD is not set.

Enabling authentication without a password would leave this gateway accepting whatever
upstream falls back to, while presenting itself as protected. Set PYZM_SERVE_AUTH_PASSWORD
(and PYZM_SERVE_AUTH_USER, which defaults to 'admin')."
    fi
}

# -----------------------------------------------------------------------------
# Environment -> upstream CLI flags
# -----------------------------------------------------------------------------
# Mapping only. Every flag below is upstream's, spelled exactly as upstream spells it; this
# function invents nothing and interprets nothing. See contracts/container-interface.md
# section 5 for the authoritative table.
build_args() {
    local -n out=$1
    out=()

    # --models is nargs="+" upstream, so a space-separated value must reach it as separate
    # argv entries. Word splitting here is deliberate, not an oversight.
    if [[ -n "${PYZM_SERVE_MODELS:-}" ]]; then
        # shellcheck disable=SC2206
        local models=(${PYZM_SERVE_MODELS})
        out+=(--models "${models[@]}")
    fi

    [[ -n "${PYZM_SERVE_BASE_PATH:-}" ]] && out+=(--base-path "${PYZM_SERVE_BASE_PATH}")
    [[ -n "${PYZM_SERVE_PROCESSOR:-}" ]] && out+=(--processor "${PYZM_SERVE_PROCESSOR}")
    [[ -n "${PYZM_SERVE_HOST:-}" ]]      && out+=(--host "${PYZM_SERVE_HOST}")
    [[ -n "${PYZM_SERVE_PORT:-}" ]]      && out+=(--port "${PYZM_SERVE_PORT}")
    [[ -n "${PYZM_SERVE_WORKERS:-}" ]]   && out+=(--workers "${PYZM_SERVE_WORKERS}")

    # Upstream store_true flags: present or absent, never valued.
    [[ -n "${PYZM_SERVE_DEBUG:-}" ]] && out+=(--debug)

    if [[ -n "${PYZM_SERVE_AUTH:-}" ]]; then
        out+=(--auth)
        [[ -n "${PYZM_SERVE_AUTH_USER:-}" ]]     && out+=(--auth-user "${PYZM_SERVE_AUTH_USER}")
        [[ -n "${PYZM_SERVE_AUTH_PASSWORD:-}" ]] && out+=(--auth-password "${PYZM_SERVE_AUTH_PASSWORD}")
        [[ -n "${PYZM_SERVE_TOKEN_SECRET:-}" ]]  && out+=(--token-secret "${PYZM_SERVE_TOKEN_SECRET}")
    fi

    return 0
}

main() {
    preflight_gpu
    preflight_auth

    local args
    build_args args

    # Trailing `docker run` arguments are appended verbatim, after ours, so an operator can
    # always reach a flag this table does not cover -- or override one it does.
    if (( $# > 0 )); then
        args+=("$@")
    fi

    # Never log the resolved command line: --auth-password and --token-secret are on it.
    log "Starting pyzm.serve (processor=${PYZM_SERVE_PROCESSOR:-gpu}, models=${PYZM_SERVE_MODELS:-yolo11m}, port=${PYZM_SERVE_PORT:-5000})"

    # exec, so the server becomes PID 1 and SIGTERM from `docker stop` reaches it directly
    # rather than being absorbed by this shell. Shutdown is clean and prompt.
    exec python3 -m pyzm.serve "${args[@]}"
}

main "$@"
