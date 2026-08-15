#!/usr/bin/env bash
#
# Pull a published image back from the registry and re-assert that it was compiled with
# CUDA (FR-010).
#
# WHY THIS IS SEPARATE FROM THE BUILD. The Dockerfile already runs the same assertion,
# twice. This one runs against the image as it exists IN THE REGISTRY, pulled fresh, in a
# job that shares nothing with the build. That independence is the entire point: it catches
# the failures that are invisible from inside a build --
#
#   - a bad BuildKit layer-cache hit that reused a stage from a different configuration
#   - a tag that ended up pointing at something other than what was just built
#   - a registry mixup, or a push that half-succeeded
#
# Running it against build-local state instead would defeat the requirement entirely.
#
# The assertion script is deliberately NOT baked into the runtime image -- the image ships
# models and a server, not verification tooling -- so it is bind-mounted in. The image's
# ENTRYPOINT is overridden because we are inspecting the image, not starting the gateway.
#
# Needs no GPU. It proves what was COMPILED, not what is AVAILABLE, which is what makes it
# runnable on a GitHub-hosted runner (FR-027). Proving the GPU actually does the work is a
# third layer that only real hardware can run -- quickstart.md Scenario 5c.
#
# Usage:
#   scripts/verify-published.sh ghcr.io/<owner>/docker-pyzm-serve:v1.2.3

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ASSERTION="assert-cuda-build.py"

usage() {
    cat >&2 <<EOF
Usage: ${0##*/} IMAGE_REFERENCE

Pulls IMAGE_REFERENCE and asserts the OpenCV inside it was compiled with CUDA.

  IMAGE_REFERENCE   A full image reference, e.g.
                    ghcr.io/<owner>/docker-pyzm-serve:v1.2.3
                    Pinning by digest (…@sha256:…) is stronger and also accepted.
EOF
    exit 2
}

main() {
    [[ $# -eq 1 ]] || usage
    local image="$1"

    [[ -f "${SCRIPT_DIR}/${ASSERTION}" ]] || {
        echo "ERROR: ${SCRIPT_DIR}/${ASSERTION} not found. Run this from a checkout." >&2
        exit 1
    }

    echo "=============================================================================="
    echo "Post-publish verification"
    echo "=============================================================================="
    echo "Image: ${image}"
    echo ""

    # --pull=always semantics: never trust a copy that happens to be in the local daemon's
    # cache, since a stale local image is one of the things this check exists to catch.
    echo "--- Pulling from the registry ---"
    docker pull --quiet "${image}"

    # Report exactly what we ended up with, so the CI log records the digest that was
    # actually verified rather than only the tag that was asked for.
    local digest
    digest="$(docker image inspect "${image}" --format '{{index .RepoDigests 0}}' 2>/dev/null || echo "(no repo digest)")"
    echo "Resolved to: ${digest}"
    echo ""

    echo "--- Asserting the pulled image was compiled with CUDA ---"
    docker run --rm \
        --entrypoint python3 \
        --volume "${SCRIPT_DIR}:/verify:ro" \
        "${image}" \
        "/verify/${ASSERTION}"

    echo ""
    echo "=============================================================================="
    echo "VERIFIED: ${image}"
    echo "  ${digest}"
    echo "=============================================================================="
    echo ""
    echo "NOTE: this proves the published image was COMPILED with CUDA. It does not, and"
    echo "cannot, prove the GPU is doing the work -- that needs real hardware. A release is"
    echo "not confirmed until quickstart.md Scenarios 4 and 5 have passed on a GPU host."
}

main "$@"
