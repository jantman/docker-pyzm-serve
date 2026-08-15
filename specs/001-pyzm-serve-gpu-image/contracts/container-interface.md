# Contract: container interface

**Feature**: `001-pyzm-serve-gpu-image` | **Date**: 2026-08-15

What operators may depend on. Everything here is a promise: changing it is a breaking change under
FR-028 and must be signalled in the version number and the release notes.

The HTTP surface is **not** defined here. It is upstream's
([`pyzm.serve` API reference](https://pyzmng.readthedocs.io/en/latest/guide/serve.html)), and
Principle III forbids this image adding endpoints or altering their semantics. This contract covers
only the container's own surface: how it is invoked, configured, observed, and identified.

---

## 1. Image identity

| | |
|---|---|
| Registry | `ghcr.io` — and nowhere else (FR-021) |
| Repository | `ghcr.io/<owner>/docker-pyzm-serve` |
| Platform | `linux/amd64` only |
| Deployable tags | `v<semver>` — immutable, the only recommended target |
| Non-deployable tags | `main-build<run_id>-<sha>` (try a `main` commit), `latest` (never recommended), `buildcache` (not a runnable image) |

Pinning by digest is stronger than pinning by tag and is always available. Release tags are
nonetheless immutable, so pinning either is safe.

---

## 2. Invocation

```bash
docker run --gpus all -p 5000:5000 ghcr.io/<owner>/docker-pyzm-serve:v<semver>
```

- **Entrypoint**: `/opt/entrypoint.sh`, which translates environment variables into
  `python -m pyzm.serve` flags and then `exec`s it as PID 1.
- **Trailing arguments**: anything after the image name is appended verbatim to the server command
  line. This is the escape hatch — the environment table below is a convenience, not a boundary.
- **Signals**: the server runs as PID 1 via `exec`, so `SIGTERM` from `docker stop` reaches it
  directly and shutdown is clean.
- **User**: runs as a non-root user. Nothing in the image needs to write to its own filesystem.

### GPU access is required by default

`--gpus all` (or an equivalent device reservation) is required unless the CPU escape hatch is set.
Without it the container **exits non-zero at start** with a diagnostic naming the missing flag. It
does not start and quietly serve CPU inference — see §6.

---

## 3. Ports and networking

| Port | Protocol | Purpose |
|---|---|---|
| `5000` | HTTP | The gateway. Configurable via `PYZM_SERVE_PORT`; `EXPOSE`d at the default. |

No outbound network access is required at any point after publication (FR-033). The image starts
and serves fully offline.

---

## 4. Volumes

**None required.** The image ships every model it serves.

| Path | Purpose | Required |
|---|---|---|
| `/var/lib/zmeventnotification/models` | Baked-in model tree | No — mount only to *add* models |

Mounting over this path replaces the shipped models rather than supplementing them, and is
therefore an operator decision with consequences. Supplying additional models is supported;
requiring the operator to supply one before the image works would violate Principle VI.

---

## 5. Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PYZM_SERVE_MODELS` | `yolo11m` | Models to load. Space-separated; `all` discovers everything (lazily loaded). |
| `PYZM_SERVE_PROCESSOR` | `gpu` | `gpu`, `cpu`, or `tpu`. |
| `PYZM_SERVE_PORT` | `5000` | Listen port. |
| `PYZM_SERVE_HOST` | `0.0.0.0` | Bind address. |
| `PYZM_SERVE_BASE_PATH` | `/var/lib/zmeventnotification/models` | Model tree root. |
| `PYZM_SERVE_WORKERS` | `1` | uvicorn workers. On GPU, more workers multiply VRAM without throughput gain. |
| `PYZM_SERVE_DEBUG` | unset | Any non-empty value enables debug logging. |
| `PYZM_SERVE_AUTH` | unset | Any non-empty value enables JWT auth. |
| `PYZM_SERVE_AUTH_USER` | `admin` | Auth username. |
| `PYZM_SERVE_AUTH_PASSWORD` | unset | Auth password. Required when auth is on. |
| `PYZM_SERVE_TOKEN_SECRET` | unset | JWT signing secret. Required when auth is on. |
| `PYZM_SERVE_ALLOW_CPU` | unset | Downgrades the GPU preflight failure to a warning. |

### Two defaults deliberately differ from upstream

| Setting | Upstream default | This image | Why |
|---|---|---|---|
| `--processor` | `cpu` | `gpu` | This is a GPU image. A user who silently gets CPU has the predecessor's bug back. |
| `--models` | `yolo11s` | `yolo11m` | `yolo11m` is the primary model per the spec; `yolo11s` remains available. |

A user following upstream's documentation will otherwise be surprised, so both are stated in the
README.

---

## 6. Startup preflight and exit codes

Before starting the server the entrypoint validates its configuration. This is not orchestration
and not an HTTP concern — it decides whether the process starts at all.

| Exit | Condition | Message names the fix |
|---|---|---|
| `0` | Normal shutdown | — |
| `1` | Server exited with an error | Upstream's message |
| `78` | GPU requested, no CUDA device visible | "…did you pass `--gpus all`?" — suppressed to a warning by `PYZM_SERVE_ALLOW_CPU=1` |
| `78` | Auth enabled without `PYZM_SERVE_TOKEN_SECRET` | Refuses to sign tokens with upstream's published `change-me` default |
| `78` | Auth enabled without `PYZM_SERVE_AUTH_PASSWORD` | Names the missing variable. An authenticated endpoint whose credentials come from someone else's source tree is worse than an open one, because the operator believes it is protected |

*Why this exists*: upstream cannot detect a CUDA-less OpenCV — `setPreferableBackend(DNN_BACKEND_CUDA)`
succeeds on a CPU-only build and inference silently falls back, with no log line to grep for
(research R6). Without this preflight, FR-004's "explicit, visible failure" would be unsatisfiable.

`78` is `EX_CONFIG` from `sysexits.h`: a configuration error, distinguishable from a crash.

---

## 7. Healthcheck

The image defines a `HEALTHCHECK` against `GET /health` on the configured port, using Python's
standard library (no `curl` in the runtime image).

- Healthy when upstream returns `{"status": "ok", "models_loaded": true}`.
- The start period accommodates model load, including first-load PTX JIT compilation on a GPU newer
  than any the image was built for — which can take tens of seconds. Too short a start period turns
  that supported edge case into a crash loop.
- With `PYZM_SERVE_MODELS=all`, weights load lazily on first request, so `models_loaded` reflects
  lazy state. The default (`yolo11m`) loads eagerly, which is what makes the healthcheck meaningful.

---

## 8. Models served

| Name a client requests | Role | Loaded by default |
|---|---|---|
| `yolo11m` | primary | yes |
| `yolo11s` | lighter alternative | no — present on disk, request it via `PYZM_SERVE_MODELS` |
| `yolov4` | legacy fallback | no — present on disk |

Names are file stems, resolved by upstream's discovery rules. `GET /models` on a running container
is the authoritative answer for what that image actually serves (SC-011).

---

## 9. GPU support

| Compute capability | Generation | Status |
|---|---|---|
| 6.1 | Pascal | Compiled |
| 7.5 | Turing | Compiled |
| 8.6 | Ampere (consumer) | Compiled |
| 8.9 | Ada | Compiled |
| > 8.9 | Blackwell and later | PTX JIT — works, with a one-off first-load delay |
| < 6.1 | Maxwell and earlier | **Not supported** — no fallback exists |

Requires an NVIDIA driver of the 550 branch or newer and the NVIDIA Container Toolkit on the host.

---

## 10. What this image does not do

Stated so nobody plans around it: no ZoneMinder API client, credentials, or event/monitor concepts;
no frame selection, zone filtering, nuisance filtering, confidence policy, past-detection matching,
or notification; no configuration file; no added or altered HTTP endpoints; no run-time model
downloads; no platform other than `linux/amd64`.

All of those belong to the client, by upstream's design and Principle III.
