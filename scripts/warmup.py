#!/usr/bin/env python3
"""Run one real inference at start-up and prove it happened on the requested processor.

WHY THIS EXISTS (Principle I, issue #1)
---------------------------------------
`entrypoint.sh`'s GPU preflight enumerates CUDA devices. Enumeration is not inference.
A container can pass that preflight, log "GPU preflight OK", report `{"status":"ok"}` on
`/health`, and then fail its very first `net.forward()` minutes later with

    (-217:Gpu API call) CUDA-capable device(s) is/are busy or unavailable in ManagedPtr

which is what happened on the deployment that produced issue #1. The allocation that
failed was `cudaMallocManaged`; enumerating devices never touches it.

The same gap makes the healthcheck's processor comparison hollow on a fresh container.
Until something has actually run a frame, `/models` reports the processor each model was
*configured* with, because nothing has had the chance to fall back yet. Principle I is
explicit that "any check that could pass while the GPU sits idle is not a check", and a
green healthcheck over an idle GPU is exactly that. This script is what puts real data
behind it, by making the first inference happen at start-up rather than whenever traffic
first arrives.

WHY IT GOES THROUGH HTTP
------------------------
The failure is per-process: a CUDA context that will not allocate in the serving process
says nothing about a context in some other process. Running a forward pass from a separate
`python3 -c` before `exec` would prove the wrong thing -- the image already does that, and
it passed while the server was degraded. Posting to the server's own `/infer` is what
exercises the process, the thread and the `cv2.dnn.Net` that will serve real requests.

WHY IT CAN KILL THE CONTAINER
-----------------------------
With `--kill-on-failure` (how `entrypoint.sh` invokes it by default), a failed warm-up sends
SIGTERM to PID 1. A start-up failure that stops the container is loud: the restart policy
recreates it, which is precisely the manual recovery the issue #1 deployment needed, and
Docker backs off if it turns into a loop. The alternative -- log and continue -- reproduces
the original bug, where the only evidence was one line nobody read.

The entrypoint withholds that flag when `PYZM_SERVE_ALLOW_CPU` is set, because that
operator has already been told the GPU may be missing and chose to start anyway; killing
the container would turn their escape hatch into a restart loop.

Note what this does NOT promise. It proves the GPU served one frame at start-up. A
transient fault that arrives later is upstream's `--gpu-retry-seconds` to heal and the
healthcheck's `processor` comparison to expose; see README "When the GPU degrades".

Stdlib only, deliberately: this runs inside the shipped image, and a warm-up that breaks
because a dependency of a dependency moved would be worse than no warm-up at all.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import sys
import time
import urllib.error
import urllib.request

# A 64x64 black JPEG. Content is irrelevant -- `blobFromImage` resizes to the model's
# input dimensions and `net.forward()` runs the same allocations whatever the pixels are,
# so the smallest valid frame is the cheapest way to exercise the real path. Embedded
# rather than generated because generating it would mean importing cv2 or PIL into this
# process purely to encode 691 bytes.
BLANK_JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcv"
    "Jik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7"
    "Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCABAAEADASIAAhEBAxEB/8QAHwAAAQUBAQEB"
    "AQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKB"
    "kaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1"
    "dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl"
    "5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcF"
    "BAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0"
    "tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDxmiiigAoo"
    "ooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q=="
)

BOUNDARY = "----pyzm-serve-warmup-boundary"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"{stamp} [warmup] {message}", flush=True)


def warn(message: str) -> None:
    log(f"WARNING: {message}")


def _get(url: str, token: str | None = None, timeout: int = 10):
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_json(url: str, payload: dict, timeout: int = 10):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _post_frame(url: str, fields: dict, jpeg: bytes, token: str | None, timeout: int):
    """POST a multipart/form-data frame to /infer.

    Hand-rolled rather than `requests` so this script keeps its stdlib-only promise.
    """
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode()
        )
    parts.append(
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="image"; filename="warmup.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n".encode()
    )
    parts.append(jpeg)
    parts.append(f"\r\n--{BOUNDARY}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={BOUNDARY}")
    req.add_header("Content-Length", str(len(body)))
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_models(base: str, deadline: float) -> bool:
    """Block until /health reports models loaded, or the deadline passes.

    Model load is slow by design on a GPU newer than anything this image compiled SASS
    for -- the PTX gets JIT'd on first load, which is why the healthcheck's start period
    is 180s. Waiting well past that is correct: a warm-up that times out while the server
    is still legitimately loading would kill a container that was about to be fine.
    """
    last_error = ""
    while time.time() < deadline:
        try:
            health = _get(f"{base}/health", timeout=8)
            if health.get("models_loaded"):
                return True
            last_error = f"models_loaded={health.get('models_loaded')!r}"
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last_error = str(exc)
        time.sleep(2)
    warn(f"gave up waiting for the server to load models: {last_error}")
    return False


def login(base: str) -> str | None:
    """Get a bearer token when auth is on; /infer is the only route that needs one."""
    if not os.environ.get("PYZM_SERVE_AUTH"):
        return None
    user = os.environ.get("PYZM_SERVE_AUTH_USER", "admin")
    password = os.environ.get("PYZM_SERVE_AUTH_PASSWORD", "")
    result = _post_json(
        f"{base}/login", {"username": user, "password": password}
    )
    return result.get("access_token")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--kill-on-failure",
        action="store_true",
        help=(
            "Send SIGTERM to PID 1 when the warm-up fails. The entrypoint passes this; "
            "run without it to diagnose a container by hand without stopping it."
        ),
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("PYZM_SERVE_WARMUP_TIMEOUT", "300")),
        help="Seconds to wait for the server to load models (default 300).",
    )
    args = ap.parse_args()

    port = os.environ.get("PYZM_SERVE_PORT", "5000")
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + args.timeout

    def fail(headline: str, detail: str) -> int:
        log("=" * 64)
        log(f"WARM-UP FAILED: {headline}")
        for line in detail.splitlines():
            log(line)
        log("")
        if args.kill_on_failure:
            log("Stopping the container. A restart policy will recreate it; if this")
            log("repeats, the GPU is not merely glitching and the log above is the")
            log("evidence to report.")
        log("=" * 64)
        if args.kill_on_failure:
            try:
                os.kill(1, signal.SIGTERM)
            except OSError as exc:  # pragma: no cover - only if PID 1 is not ours
                warn(f"could not signal PID 1: {exc}")
        return 1

    if not wait_for_models(base, deadline):
        return fail(
            "the server never reported its models as loaded",
            f"Waited {args.timeout}s for {base}/health to return models_loaded=true.\n"
            "Nothing was inferred, so nothing about the GPU has been proven.",
        )

    try:
        models = _get(f"{base}/models", timeout=10).get("models", [])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return fail("could not read /models", str(exc))

    if not models:
        # PYZM_SERVE_MODELS=all loads lazily: nothing is resolved until a request names a
        # model, so there is nothing here to warm up. Say so rather than passing silently,
        # because "warm-up OK" would otherwise imply a proof that was never run.
        warn("no models are loaded (lazy mode?), so there is nothing to warm up.")
        warn("The GPU remains UNPROVEN until the first real request.")
        return 0

    try:
        token = login(base)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return fail(
            "could not log in to run the warm-up",
            f"{exc}\nPYZM_SERVE_AUTH is set, so /infer needs a token. Check "
            "PYZM_SERVE_AUTH_USER and PYZM_SERVE_AUTH_PASSWORD.",
        )

    for model in models:
        name = model.get("name", "")
        fields = {"type": model.get("type", "object"), "name": name}
        started = time.time()
        try:
            result = _post_frame(
                f"{base}/infer", fields, BLANK_JPEG, token, timeout=max(60, args.timeout)
            )
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return fail(f"inference request for {name!r} did not complete", str(exc))
        elapsed = (time.time() - started) * 1000

        # Upstream answers a failed inference with 200 and an `error` key rather than an
        # HTTP error, including under --no-cpu-fallback. Ignoring the body would make this
        # warm-up report success for a frame that was never inferred.
        if result.get("error"):
            return fail(
                f"the server refused to infer with {name!r}",
                f"{result['error']}\n"
                "With --no-cpu-fallback this is what a GPU failure looks like: the model "
                "stays on the GPU and the request fails, rather than degrading.",
            )
        log(f"{name}: warm-up inference OK in {elapsed:.0f} ms")

    # The inference above is what makes this comparison mean anything: `processor` can
    # only have moved off `requested_processor` once something has actually run.
    try:
        after = _get(f"{base}/models", timeout=10).get("models", [])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return fail("could not re-read /models after the warm-up", str(exc))

    unverifiable = [m for m in after if not m.get("processor")]
    if unverifiable:
        # Pre-#67 upstream has no `processor` key. The frame still ran, so the warm-up has
        # done its main job; it just cannot say what ran it.
        warn(
            "this pyzm does not report `processor` on /models, so the warm-up proved "
            "that inference works but NOT which processor ran it."
        )
        return 0

    degraded = [
        m for m in after if m.get("processor") != m.get("requested_processor")
    ]
    if degraded:
        detail = "\n".join(
            f"  {m.get('name')}: running on {m.get('processor')}, "
            f"requested {m.get('requested_processor')}"
            for m in degraded
        )
        return fail(
            "the first inference fell back off the requested processor",
            f"{detail}\n\n"
            "The frame was inferred, but not by the processor you asked for. Serving "
            "this way is the silent-CPU failure this image exists to prevent, so the "
            "container is stopping instead of answering requests several times slower.\n"
            "Set PYZM_SERVE_WARMUP=0 to start anyway; upstream will retry the GPU on its "
            "own schedule and the healthcheck will report the container unhealthy "
            "meanwhile.",
        )

    processors = ", ".join(
        f"{m.get('name')}={m.get('processor')}" for m in after
    )
    log(f"warm-up complete: {processors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
