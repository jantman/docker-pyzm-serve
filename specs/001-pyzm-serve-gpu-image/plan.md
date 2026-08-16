# Implementation Plan: GPU-Accelerated pyzm.serve Inference Gateway Image

**Branch**: `main` | **Date**: 2026-08-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-pyzm-serve-gpu-image/spec.md`

## Summary

Build the whole project from an empty repository: a multi-stage container image that serves
`python -m pyzm.serve` on an NVIDIA GPU, and the two GitHub Actions workflows that publish and
verify it to GHCR.

The technical approach is set by two research findings ([research.md](./research.md)):

1. **The `opencv-python` shadowing hazard is avoidable, not merely survivable.** `pyzm[serve]`
   declares neither Ultralytics nor `opencv-python` — only `[train]` and `[full]` do. Installing the
   right extra removes the hazard at source; Ultralytics is confined to a throwaway model-export
   stage whose site-packages never reach the shipped image; upstream's own dist-info shim is adopted
   as defence in depth against a future transitive dependency.
2. **Upstream cannot detect a CUDA-less OpenCV**, so the log-grep verification inherited from the
   consuming project's contract passes in exactly the failure case it exists to catch. Verification
   is therefore a three-layer ladder — a build-time assertion over `cv2.getBuildInformation()`, a
   post-publish re-assertion against the image *pulled back from the registry*, and a documented
   runtime check the operator runs on real hardware — plus an entrypoint preflight that refuses to
   start when GPU inference was requested and no CUDA device is visible.

Everything is baked in: OpenCV 4.12.0 compiled from source with CUDA and cuDNN for four GPU
generations, YOLO11m/YOLO11s exported to ONNX at build time, and YOLOv4 Darknet weights retained as
the legacy fallback. The image starts and serves with no network and no volumes.

## Technical Context

**Language/Version**: Python 3.10 (Ubuntu 22.04 system Python, satisfying pyzm's `>=3.10`); C++17
for the OpenCV compile; Bash for the entrypoint; YAML for workflows. No application code of our own
beyond the entrypoint and verification scripts — Principle III forbids more.

**Primary Dependencies**: `pyzm[serve]` 2.5.1 (FastAPI + uvicorn + ONNX); OpenCV 4.12.0 +
`opencv_contrib` built from source with `WITH_CUDA=ON`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON`; CUDA
12.4.1 + cuDNN from `nvidia/cuda` base images; Ultralytics (build-stage only, for ONNX export).

**Storage**: None. Models are read-only files baked into the image at
`/var/lib/zmeventnotification/models`. No database, no state, no writable volume required.

**Testing**: No unit-test suite — there is no application logic to unit-test. Verification is
executable and layered: a build-time `RUN` assertion that fails the build, a post-publish CI job
that re-asserts against the pulled image, and a documented operator runtime check. See
[quickstart.md](./quickstart.md).

**Target Platform**: `linux/amd64` containers on a host with an NVIDIA GPU of compute capability
6.1/7.5/8.6/8.9 (newer JITs via PTX), NVIDIA driver ≥ 550, and the NVIDIA Container Toolkit.

**Project Type**: Container image + CI/CD. Single deliverable, no source tree in the conventional
sense.

**Performance Goals**: single-frame object detection at ≤ 300 ms p95 on Pascal-class hardware
(SC-001), against the ~800 ms CPU baseline of the image this replaces. One uvicorn worker is
correct on GPU; extra workers multiply VRAM without throughput gain.

**Constraints**: GitHub-hosted runners cap a job at 6 hours and larger runners need a paid plan the
maintainer does not have — the OpenCV CUDA compile across four architectures is the only step at
risk, mitigated by a trimmed `BUILD_LIST`, registry-backed BuildKit cache, and ccache (research R8).
The shipped image must contain no compiler toolchain (FR-018) and must start with no network and no
volumes (FR-033/FR-034).

**Scale/Scope**: One image, two workflows, one entrypoint, three model artifacts, one operator.
Concurrency is one home security system's event rate — single-digit requests per second at peak.

## Constitution Check

*GATE: evaluated before Phase 0 research, re-evaluated after Phase 1 design.*

Constitution v1.2.0. Both evaluations below are recorded; nothing changed between them except where
noted.

*Originally evaluated against v1.1.0. Implementation of Principle III produced amendment v1.2.0 —
see the Principle III row for what changed and why. No other principle's evaluation is affected.*

| Principle | Gate | Initial | Post-design |
|---|---|---|---|
| **I. Prove the GPU Path** | Build fails without compiled-CUDA evidence; CI re-verifies the published artifact; README documents the runtime check; no check may pass with the GPU idle | PASS | **PASS — strengthened.** R6 showed the contract's log-grep check passes in the failure case, so the ladder in R6 replaced it. The entrypoint preflight (R7) was added because upstream cannot fail loudly on its own. |
| **II. Pin Everything** | Base image by digest; upstream checkouts by full SHA via ARG; models by version + checksum; toolchain-relevant packages pinned | PASS | **PASS.** Both base digests captured (R1). `yolov4.cfg`/`coco.names` moved from a floating `master` ref to vendored files — a pin the naive approach would have missed. Honest limit recorded: the exported `.onnx` is not bit-reproducible, so inputs are gated and the output checksum is observed. |
| **III. Dumb Inference Engine** | No ZoneMinder awareness of our own; no orchestration; no config file; upstream's HTTP surface unaltered | PASS | **PASS — principle amended.** Env vars map to upstream CLI flags and nothing more. The entrypoint preflight is not an endpoint and does not alter request semantics — it decides whether the process starts. Upstream's `--config` YAML exists but we neither ship nor document one. `/openapi.json` on a running container exposes exactly upstream's four routes. Implementation found the original "no ZoneMinder awareness" bullet unsatisfiable — `pyzm.serve.app` transitively imports `pyzm.zm.*` and `pyzm.client` — so it was rescoped to what this repository *adds*, with a behavioural test, in **Constitution v1.2.0**. The delivered image satisfies it: no ZM host, credential or URL variable is declared, and it serves inference under `--network none`. |
| **IV. Immutable Promises** | Tag equals git tag; never overwrite; breaking changes signalled; GHCR only; OCI labels + SBOM; `main` builds under a distinct tag | PASS | **PASS.** One deliberate wrinkle: the BuildKit `:buildcache` tag is mutable. It is not a release and not a runnable image; named and documented so it cannot be mistaken for one (R8/R9). |
| **V. Public, Personal, Best-Effort** | Nothing author-specific; README sufficient for a stranger; sensible general defaults; status badge and honest support framing | PASS | **PASS.** Four GPU generations, not one. `PYZM_SERVE_ALLOW_CPU` exists so a GPU-less user is not refused outright. Every site-specific value is an env var with a general default. |
| **VI. Self-Contained by Construction** | Everything baked in; checksums at build; starts with no network and no volumes; same tag ⇒ same environment | PASS | **PASS.** Models exported and baked at build. The no-network/no-volume start is an executable check in quickstart.md, not an aspiration. |

**Build & Runtime Constraints**: OpenCV from source ✓; multiple CUDA architectures including 6.1 as
an explicit ARG ✓; install-order rationale commented in the Dockerfile ✓ (FR-009); base image pinned
✓; expensive stage cached and ordered first ✓; multi-stage shipping runtime artifacts only ✓;
`linux/amd64` ✓; documented model set ✓; `HEALTHCHECK` against `/health` ✓.

**Development Workflow**: work lands on `main` ✓; green before done ✓; real-hardware verification
before release is an explicit gate, not CI's job ✓; the consuming project's contract is treated as
one integration check, not the acceptance surface ✓.

**Result: PASS, no violations.** Complexity Tracking is therefore omitted.

One consequence worth stating rather than burying: the consuming project's contract verification
block is **kept and extended, not adopted as-is**. Its log-grep step is retained because it catches
genuine runtime CUDA errors, but it is documented as insufficient on its own. That is Development
Workflow §4 in practice — the contract is a useful integration check, not this project's definition
of correct.

## Project Structure

### Documentation (this feature)

```text
specs/001-pyzm-serve-gpu-image/
├── plan.md              # This file
├── research.md          # Phase 0 output — R1..R12, incl. the two headline findings
├── data-model.md        # Phase 1 output — entities, pins, tags, config surface
├── quickstart.md        # Phase 1 output — executable verification scenarios
├── contracts/
│   ├── container-interface.md   # What operators depend on: args, env, ports, exit codes, tags
│   └── cicd-contract.md         # What the pipeline promises: triggers, tags, verification, perms
├── checklists/
│   └── requirements.md  # Spec quality checklist (complete)
└── tasks.md             # Phase 2 output (/speckit-tasks — NOT created here)
```

### Source Code (repository root)

```text
.
├── Dockerfile                  # Multi-stage: opencv-build → model-export → runtime
├── docker-compose.yml          # Worked example with GPU reservation (FR-031)
├── entrypoint.sh               # Env → upstream CLI flags; GPU preflight (R7)
├── .dockerignore               # Keeps doc churn and downloaded model binaries out of the context
├── .gitignore                  # Keeps those same downloaded binaries out of git
├── README.md                   # Badge, honest framing, quickstart, GPU verification (FR-029/030)
├── LICENSE                     # Already present
├── models/
│   ├── checksums.sha256        # Pinned input checksums, verified at build (FR-014)
│   ├── yolov4.cfg              # Vendored — was a floating master ref
│   └── coco.names              # Vendored — YOLOv4 labels (YOLO11 reads its own metadata)
├── scripts/
│   ├── assert-cuda-build.py    # Build + CI assertion: compiled with CUDA (FR-008/009/010)
│   ├── export-onnx.py          # Build-stage only: .pt → .onnx at imgsz=640
│   └── verify-published.sh     # CI: pull from registry, re-run the assertion
└── .github/workflows/
    ├── build.yml               # push to main + workflow_dispatch → non-release tag
    └── release.yml             # push tag → immutable tag, verify, GitHub release
```

**Structure Decision**: A flat repository root, matching `docker-zm-mlapi`, because the deliverable
is one image rather than an application. There is no `src/` or `tests/`: Principle III caps our own
code at an entrypoint and two verification scripts, and there is no application logic to unit-test.
`scripts/assert-cuda-build.py` is deliberately a standalone file rather than an inline `RUN` string
so that the identical assertion executes at build time and again in CI against the pulled image —
FR-010's independence is worth nothing if the two checks can drift apart.
