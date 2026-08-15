#!/usr/bin/env python3
"""Assert that the OpenCV importable from this interpreter was COMPILED with CUDA.

This is the single gate Constitution Principle I hangs on, and it is executed by three
different callers so they cannot drift apart:

  1. the Dockerfile, twice -- once against the freshly built OpenCV in the compile stage,
     and again in the runtime stage after the opencv-python shim and pyzm are installed,
     so anything those steps could have disturbed is caught (FR-008, FR-009);
  2. scripts/verify-published.sh, run by CI against the image *pulled back from the
     registry* -- which catches a bad layer-cache hit, a wrong tag, or a registry mixup
     that is invisible from inside the build (FR-010);
  3. quickstart.md Scenario 5a, run by an operator.

It deliberately DOES NOT REQUIRE A GPU. It tests what was *compiled*, not what is
*available*, which is what makes it usable on a GPU-less CI runner (FR-027). Proving that
the GPU is actually doing the work is a separate, third layer that only real hardware can
run -- see quickstart.md Scenario 5c.

Why this check and not a log grep: upstream cannot detect a CUDA-less OpenCV.
`YoloBase._setup_gpu()` (pyzmNg v2.5.1, pyzm/ml/backends/yolo.py:204-231) checks the OpenCV
*version*, not its capabilities, and `setPreferableBackend(DNN_BACKEND_CUDA)` succeeds on a
CPU-only build and then silently falls back at forward() time. So the obvious check --
grepping the container log for "fell back" -- passes in exactly the failure case it exists
to catch (research R6). That failure ran a production stack on the CPU for over a year.

Exit codes:
    0  OpenCV was compiled with CUDA and the cv2.cuda bindings are present.
    1  It was not, or cv2 could not be imported at all.
"""

import re
import sys

BANNER = "=" * 78


def fail(message, detail=None):
    """Print an explicit, actionable failure and exit non-zero."""
    print(BANNER, file=sys.stderr)
    print("CUDA BUILD ASSERTION FAILED", file=sys.stderr)
    print(BANNER, file=sys.stderr)
    print(message, file=sys.stderr)
    if detail:
        print("", file=sys.stderr)
        print(detail, file=sys.stderr)
    print("", file=sys.stderr)
    print(
        "This image must never ship an OpenCV that cannot use the GPU: it would serve\n"
        "CPU inference while reporting processor:gpu, with nothing in the logs to say so.\n"
        "See Constitution Principle I and specs/001-pyzm-serve-gpu-image/research.md R6.",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    # --- 1. cv2 must import at all -----------------------------------------------
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001 - any import failure is fatal here
        fail(
            "Could not import cv2.",
            "Underlying error: {}: {}".format(type(exc).__name__, exc),
        )

    build_info = cv2.getBuildInformation()

    # --- 2. The build must report NVIDIA CUDA: YES --------------------------------
    #
    # A CUDA-less build either reports "NVIDIA CUDA: NO" or omits the line entirely,
    # depending on how the configure step failed. Both are failures; treating a missing
    # line as "probably fine" is precisely the assumption this file exists to refuse.
    cuda_lines = [ln.rstrip() for ln in build_info.splitlines() if "CUDA" in ln.upper()]
    match = re.search(r"^\s*NVIDIA CUDA:\s*(\S+)(.*)$", build_info, re.MULTILINE)

    if match is None:
        fail(
            "cv2.getBuildInformation() contains no 'NVIDIA CUDA:' line at all, which means "
            "this OpenCV was configured without CUDA support.",
            "CUDA-mentioning lines found in the build information:\n  "
            + ("\n  ".join(cuda_lines) if cuda_lines else "(none)"),
        )

    verdict = match.group(1).upper()
    if verdict != "YES":
        fail(
            "cv2.getBuildInformation() reports 'NVIDIA CUDA: {}' -- expected YES.".format(
                match.group(1)
            ),
            "CUDA-mentioning lines found in the build information:\n  "
            + "\n  ".join(cuda_lines),
        )

    # --- 3. The cv2.cuda namespace must exist -------------------------------------
    #
    # This is a distinct claim from the one above: WITH_CUDA=ON without opencv_contrib
    # produces a cv2 with no cuda namespace, because cudaarithm -- which is what puts
    # cv2.cuda into the Python bindings -- lives in contrib. Without it the runtime check
    # in quickstart Scenario 5b (getCudaEnabledDeviceCount) cannot even be called, and the
    # entrypoint GPU preflight has nothing to preflight with.
    if not hasattr(cv2, "cuda"):
        fail(
            "OpenCV reports CUDA support but the cv2.cuda namespace is missing.",
            "This means the contrib modules were not built. cv2.cuda comes from "
            "cudaarithm in opencv_contrib; OPENCV_EXTRA_MODULES_PATH and the cuda* entries "
            "in BUILD_LIST are what provide it. Without it, neither the entrypoint GPU "
            "preflight nor quickstart Scenario 5b can run.",
        )

    if not hasattr(cv2.cuda, "getCudaEnabledDeviceCount"):
        fail(
            "cv2.cuda exists but cv2.cuda.getCudaEnabledDeviceCount is missing.",
            "The entrypoint GPU preflight calls exactly this function; without it the "
            "container cannot tell whether a GPU is visible and FR-004 is unsatisfiable.",
        )

    # --- 4. Report what we proved -------------------------------------------------
    print(BANNER)
    print("CUDA BUILD ASSERTION PASSED")
    print(BANNER)
    print("OpenCV version: {}".format(cv2.__version__))
    print("")
    print("CUDA-related build configuration:")
    for line in cuda_lines:
        print("  {}".format(line))
    print("")

    # Deliberately reported, never asserted on: this file must pass on a GPU-less runner.
    # A device count of 0 here is expected in CI and means nothing about the build.
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
    except Exception as exc:  # noqa: BLE001 - informational only
        count = "unavailable ({}: {})".format(type(exc).__name__, exc)
    print("Visible CUDA devices at this moment: {}".format(count))
    print(
        "  (informational only -- this assertion is about what was COMPILED. Zero is the\n"
        "   correct, expected answer on a build runner. Proving the GPU is actually doing\n"
        "   the work needs real hardware: quickstart.md Scenario 5b and 5c.)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
