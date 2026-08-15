# Data Model: GPU-Accelerated pyzm.serve Inference Gateway Image

**Feature**: `001-pyzm-serve-gpu-image` | **Date**: 2026-08-15

This project has no database and no persistent state. The entities below are build- and
deploy-time artifacts: the things that get pinned, produced, published and depended upon. They are
modelled here because the invariants between them are what FR-014, FR-016, FR-024, FR-033 and
FR-034 actually constrain.

Detection payloads are **not** modelled here. They belong to upstream's HTTP interface, which
Principle III forbids us altering; see [contracts/container-interface.md](./contracts/container-interface.md)
for the pointer to upstream's schema.

---

## Entity: Pinned Input

Anything external the image is built from. Every instance carries an exact immutable identifier.

| Field | Type | Notes |
|---|---|---|
| `kind` | enum | `base-image`, `git-checkout`, `model-weights`, `pypi-package`, `vendored-file` |
| `identifier` | string | Digest, full commit SHA, release-asset URL, or repo-relative path |
| `verification` | string | Digest match, SHA-256 checksum, or "in-tree" for vendored files |
| `build_arg` | string? | The Dockerfile `ARG` exposing it, where overridable |

**Instances**:

| Input | Kind | Identifier | Build ARG |
|---|---|---|---|
| CUDA devel base | base-image | `nvidia/cuda@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42` (`12.4.1-cudnn-devel-ubuntu22.04`) | `CUDA_DEVEL_IMAGE` |
| CUDA runtime base | base-image | `nvidia/cuda@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6` (`12.4.1-cudnn-runtime-ubuntu22.04`) | `CUDA_RUNTIME_IMAGE` |
| OpenCV | git-checkout | tag `4.12.0`, resolved to a full SHA | `OPENCV_REF` |
| opencv_contrib | git-checkout | tag `4.12.0`, resolved to a full SHA | `OPENCV_CONTRIB_REF` |
| pyzmNg | git-checkout | tag `v2.5.1`, resolved to a full SHA | `PYZM_REF` |
| `yolo11m.pt` | model-weights | `ultralytics/assets` release `v8.4.0` asset + SHA-256 | `YOLO11_ASSETS_REF` |
| `yolo11s.pt` | model-weights | same release + SHA-256 | `YOLO11_ASSETS_REF` |
| `yolov4.weights` | model-weights | `AlexeyAB/darknet` release `darknet_yolo_v3_optimal` + SHA-256 | `YOLOV4_WEIGHTS_URL` |
| `ultralytics` | pypi-package | exact `==` version (export stage only) | `ULTRALYTICS_VERSION` |
| `torch` (CPU wheel) | pypi-package | exact `==` version (export stage only) | `TORCH_VERSION` |
| `yolov4.cfg` | vendored-file | `models/yolov4.cfg` | — |
| `coco.names` | vendored-file | `models/coco.names` | — |

**Validation rules**:

- **PI-1**: No identifier may be a floating reference — no `latest`, no branch name, no unversioned
  URL. (FR-016)
- **PI-2**: Every `model-weights` input is checksum-verified during the build; a mismatch fails the
  build rather than warning. (FR-014)
- **PI-3**: Checksums live in `models/checksums.sha256`, in git, reviewable in a diff.
- **PI-4**: The two vendored files exist as in-tree files precisely because their upstream homes
  are branch refs, which PI-1 forbids.

---

## Entity: Model Artifact

A detection model present in the shipped image. Baked in at build time; never fetched at run time.

| Field | Type | Notes |
|---|---|---|
| `name` | string | The name a client requests; derived from the file stem |
| `role` | enum | `primary`, `alternative`, `legacy-fallback` |
| `path` | path | Under `/var/lib/zmeventnotification/models/` |
| `framework` | enum | `opencv-onnx`, `opencv-darknet` — inferred by upstream from the extension |
| `loaded_by_default` | bool | True only for the primary model |
| `labels_source` | enum | `onnx-metadata` or `labels-file` |
| `provenance` | ref | The Pinned Input it derives from |

**Instances**:

| Name | Role | Path | Framework | Default | Labels |
|---|---|---|---|---|---|
| `yolo11m` | primary | `ultralytics/yolo11m.onnx` | opencv-onnx | yes | onnx-metadata |
| `yolo11s` | alternative | `ultralytics/yolo11s.onnx` | opencv-onnx | no | onnx-metadata |
| `yolov4` | legacy-fallback | `yolov4/yolov4.weights` (+ `.cfg`, `coco.names`) | opencv-darknet | no | labels-file |

**Validation rules**:

- **MA-1**: Exactly one artifact has `role: primary` and `loaded_by_default: true`.
- **MA-2**: Non-default artifacts are present on disk but unnamed in the default `--models`, so they
  do not appear in `GET /models` until an operator asks for them. "Present but not loaded" (FR-013)
  means present on disk, not registered-and-idle.
- **MA-3**: ONNX artifacts are exported at `imgsz=640`, matching `yolo_onnx.py:25`'s
  `_DEFAULT_DIM`. An artifact exported at another size would be silently mis-scaled.
- **MA-4**: ONNX artifacts are exported with `nms=False`, keeping upstream on its well-trodden
  parsing path rather than the end-to-end fallback branch.
- **MA-5**: The Darknet artifact requires all three files co-located; upstream discovers `.cfg` and
  labels from the same directory.
- **MA-6**: Names are file stems. Renaming a file renames the model a client must request.

**State**: models have no lifecycle within a container — read-only, loaded once at start (primary)
or on first request (if the operator selects `--models all`).

---

## Entity: Published Image

One immutable artifact per publish.

| Field | Type | Notes |
|---|---|---|
| `registry` | const | `ghcr.io` — the only registry (FR-021) |
| `repository` | string | `ghcr.io/<owner>/docker-pyzm-serve` |
| `tag` | string | See Release Tag below |
| `digest` | string | The identifier users should ultimately pin |
| `labels` | map | OCI `url`, `source`, `version`, `revision` (FR-025) |
| `sbom` | attestation | Attached at build (FR-025) |
| `provenance` | attestation | `mode=max` |
| `platform` | const | `linux/amd64` (FR-020) |
| `verified` | bool | Set only after the pull-back assertion passes (FR-010) |

**Validation rules**:

- **PIm-1**: A published tag is never overwritten, moved, or deleted. (FR-024)
- **PIm-2**: An image whose pull-back verification did not run or did not pass is not presented as
  usable. (FR-026)
- **PIm-3**: Contains no compiler toolchain. (FR-018)
- **PIm-4**: Two containers from the same tag have identical model sets and checksums. (FR-034)

---

## Entity: Release Tag

The git tag and the identically-named registry tag it produces.

| Field | Type | Notes |
|---|---|---|
| `git_tag` | string | e.g. `v0.1.0` |
| `image_tag` | string | Equal to `git_tag` (FR-023) |
| `github_release` | ref | Created by the release workflow |
| `breaking` | bool | Signalled in version and release notes (FR-028) |

**Tag namespace** — three kinds, deliberately unmistakable for one another:

| Pattern | Kind | Mutable | Deployable | Purpose |
|---|---|---|---|---|
| `v<semver>` | release | **no** | **yes** — the only recommended target | The promise |
| `main-build<run_id>-<sha>` | development | no | not recommended | Try a `main` commit (FR-022) |
| `latest` | rolling | yes | **no** — never recommended (FR-024) | Convenience only |
| `buildcache` | cache | yes | **not an image** | BuildKit layer cache (R8) |

**Validation rules**:

- **RT-1**: A development tag can never collide with a release tag — the `main-build` prefix and the
  run ID make the namespaces disjoint.
- **RT-2**: `buildcache` holds cache manifests, not a runnable image. It is mutable by design and is
  the one deliberate exception to the immutability habit; being neither a release nor runnable, it
  does not engage FR-024.
- **RT-3**: A breaking change (removed model, changed default, dropped GPU generation, new required
  setting) requires a version bump that communicates it.

---

## Entity: Runtime Configuration

Operator-supplied settings. Environment variables only — no configuration file (Principle III).

| Variable | Default | Maps to | Notes |
|---|---|---|---|
| `PYZM_SERVE_MODELS` | `yolo11m` | `--models` | Upstream default is `yolo11s`; ours differs |
| `PYZM_SERVE_PROCESSOR` | `gpu` | `--processor` | Upstream default is `cpu`; ours differs |
| `PYZM_SERVE_PORT` | `5000` | `--port` | |
| `PYZM_SERVE_HOST` | `0.0.0.0` | `--host` | |
| `PYZM_SERVE_BASE_PATH` | `/var/lib/zmeventnotification/models` | `--base-path` | |
| `PYZM_SERVE_WORKERS` | `1` | `--workers` | >1 multiplies VRAM without GPU gain |
| `PYZM_SERVE_DEBUG` | unset | `--debug` | |
| `PYZM_SERVE_AUTH` | unset (off) | `--auth` | |
| `PYZM_SERVE_AUTH_USER` | `admin` | `--auth-user` | Only with auth on |
| `PYZM_SERVE_AUTH_PASSWORD` | unset | `--auth-password` | Required when auth on |
| `PYZM_SERVE_TOKEN_SECRET` | unset | `--token-secret` | Required when auth on |
| `PYZM_SERVE_ALLOW_CPU` | unset | — | Downgrades the GPU preflight to a warning |

**Validation rules**:

- **RC-1**: Two defaults deliberately diverge from upstream (`--models`, `--processor`). Both must
  be stated in the README, because a user following upstream's docs will otherwise be surprised.
  (FR-032's sibling concern: no surprises, no hidden site-specific tuning.)
- **RC-2**: With `PYZM_SERVE_PROCESSOR=gpu` and no visible CUDA device, the entrypoint exits
  non-zero with a diagnostic naming `--gpus all` — unless `PYZM_SERVE_ALLOW_CPU=1`. (FR-004, R7)
- **RC-3**: With auth enabled and either `PYZM_SERVE_TOKEN_SECRET` or `PYZM_SERVE_AUTH_PASSWORD`
  unset, the entrypoint refuses to start, naming the missing variable — rather than signing tokens
  with upstream's published `change-me` default or accepting whatever password upstream falls back
  to. Both are marked required-when-auth-on in the container contract and neither has a default
  here. (FR-007)
- **RC-4**: Arguments passed after the image name are appended to the server command line, so the
  table is a convenience, never a cage.
- **RC-5**: No variable may carry an author-specific default. (FR-032)

---

## Relationships

```text
Pinned Input ──(build, checksum-verified)──> Model Artifact
Pinned Input ──(build)──────────────────────> Published Image
Model Artifact ──(baked into)───────────────> Published Image
Published Image ──(tagged as)───────────────> Release Tag
Release Tag ────(pinned by)─────────────────> Operator deployment
Runtime Configuration ──(applied at start)──> Published Image instance
```

The invariant that ties them together, and the one Principle VI exists to protect: **every arrow
above is traversed at build time except the last.** Nothing about which models exist, which
versions they are, or what the environment contains is decided after publication.
