# Phase 0 Research: GPU-Accelerated pyzm.serve Inference Gateway Image

**Feature**: `001-pyzm-serve-gpu-image` | **Date**: 2026-08-15

All sources were checked against upstream at tag `v2.5.1` (`ZoneMinder/pyzmNg`) or against live
registry/API data on the date above. Where a fact came from reading upstream source rather than
documentation, the file and line are cited, because two of the findings below contradict what the
documentation implies.

---

## Headline findings

Two results change the shape of the implementation and are worth stating before the detail.

**1. `pyzm[serve]` does not depend on Ultralytics, and therefore not on `opencv-python`.** The
build hazard inherited from the consuming project's contract — "Ultralytics pulls `opencv-python`
from PyPI, which shadows a source-built OpenCV" — is real, but it applies to the `[train]` and
`[full]` extras, not to `[serve]`. Installing the correct extra removes the hazard at its source
rather than working around it. See R4.

**2. Upstream cannot detect a CUDA-less OpenCV, so the contract's log-grep verification passes in
exactly the failure case it exists to catch.** This is not a criticism of upstream — the client is
not where that check belongs — but it means the verification block in the consuming project's
contract is necessary and insufficient, and that FR-004 cannot be satisfied by configuration
alone. See R6, which is the most important entry here.

---

## R1. Base image

**Decision**: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` for build stages,
`nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` for the shipped stage, both pinned by digest.

| Tag | Digest | Compressed size |
|---|---|---|
| `12.4.1-cudnn-devel-ubuntu22.04` | `sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42` | 4.24 GB |
| `12.4.1-cudnn-runtime-ubuntu22.04` | `sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6` | 2.07 GB |

**Rationale**: The `-cudnn-` variant is not optional. OpenCV's DNN CUDA backend
(`OPENCV_DNN_CUDA=ON`) requires cuDNN at both build and run time, and `.onnx` models are served
through OpenCV DNN (R5), so cuDNN is on the hot path. Taking it from NVIDIA's own image avoids
pinning a second apt repository. The `runtime` variant carries the CUDA and cuDNN shared libraries
without `nvcc`, which is exactly what FR-018 asks for.

Ubuntu 22.04 rather than 20.04 because it ships Python 3.10, and `pyzm` requires `>=3.10`
(`setup.py`, `python_requires`). 22.04 also predates PEP 668's externally-managed marker, so
`pip install` works without a venv or `--break-system-packages` — one less moving part than the
sibling repository needs on Debian 12.

**Alternatives considered**:

- *Debian + NVIDIA's CUDA apt repository* — keeps parity with `docker-zm-mlapi`, but adds a
  repository, a GPG key and a set of unpinned package versions, all of which are pinning surfaces
  under Principle II. Rejected in the spec; recorded in the Constitution as settled.
- *Ubuntu 24.04 CUDA images* — not published for CUDA 12.4.1; the family starts at 12.6. Adopting
  one means moving off the CUDA generation the consuming host's 550 driver is matched to.
- *Non-cuDNN `12.4.1-devel`* — smaller, but `OPENCV_DNN_CUDA=ON` will not configure without cuDNN,
  and installing it separately means pinning NVIDIA's apt repository after all.

**Known cost**: Ubuntu 22.04 standard support ends April 2027. The base image is an `ARG`; moving
to a newer CUDA generation is a normal change gated by the same verification as everything else.

---

## R2. OpenCV version and build configuration

**Decision**: OpenCV **4.12.0** plus `opencv_contrib` 4.12.0, both from pinned git tags, built with:

```
-D WITH_CUDA=ON
-D WITH_CUDNN=ON
-D OPENCV_DNN_CUDA=ON
-D WITH_CUBLAS=ON
-D CUDA_ARCH_BIN=<see R3>
-D OPENCV_EXTRA_MODULES_PATH=<opencv_contrib>/modules
-D BUILD_LIST=<see below>
-D CMAKE_BUILD_TYPE=Release
-D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_DOCS=OFF
-D BUILD_opencv_apps=OFF -D BUILD_JAVA=OFF
-D OPENCV_GENERATE_PKGCONFIG=ON
-D BUILD_opencv_python3=ON
```

`opencv_contrib` is **required**, not optional: `OPENCV_DNN_CUDA=ON` depends on the `cudev` module,
which lives in contrib. A build that omits contrib silently produces a CUDA-less DNN — the failure
mode this whole project exists to prevent.

**Version rationale**: 4.14.0 is the current 4.x release (July 2026) and 5.0.0 shipped in June 2026,
so 4.12.0 is a deliberate step back. The reason is version skew. OpenCV's CUDA code is sensitive to
the CUDA toolkit it is compiled against in both directions: 4.9 fails to compile against cuDNN 9
(opencv#24983), and 4.12 has reported `cub` failures against CUDA 12.9
(forum.opencv.org/t/23451). We are pinned to CUDA 12.4 (April 2024) by the consuming host's driver,
and 4.12.0 (July 2025) is the most recent release whose CUDA-12.4-era pairing is well trodden.
4.10.0 is the floor — it is the first release with working cuDNN 9 support.

This is a build `ARG`. Moving to 4.13/4.14 is a one-line change whose correctness is decided by the
CUDA assertion (R6) and a smoke inference, not by opinion.

**`BUILD_LIST` rationale**: compiling every module for four GPU architectures is what puts this
build near the runner's wall (R8). The list is the primary lever, so it is set deliberately:

```
core,imgproc,imgcodecs,videoio,dnn,python3,cudev,cudaarithm,cudawarping,cudaimgproc
```

`dnn` is the engine; `imgcodecs` decodes the uploaded PNG; `imgproc` does the letterbox resize
(`yolo_onnx.py:165`); `cudev` is DNN-CUDA's dependency; `cudaarithm` is what puts `cv2.cuda` — and
therefore `getCudaEnabledDeviceCount()` — into the Python bindings, making it load-bearing for
verification rather than incidental. `videoio` is included as cheap insurance against an upstream
import; it can be dropped if build time demands it, but only with a passing smoke test.

**Alternatives considered**: `opencv-python-cuda-wheels` (cudawarped) publishes prebuilt CUDA
wheels and would remove the compile entirely. Rejected: it is a third-party republisher, its builds
target specific CUDA/arch combinations we do not control, and Principle I wants the build to be the
thing we assert over. Reconsider only if build time becomes untenable.

---

## R3. CUDA architectures

**Decision**: `CUDA_ARCH_BIN=6.1;7.5;8.6;8.9` with PTX emitted for 8.9, exposed as a build `ARG`.

| Capability | Generation | Representative hardware |
|---|---|---|
| 6.1 | Pascal | GTX 10-series, Quadro P1000 — required by FR-017 |
| 7.5 | Turing | GTX 16-series, RTX 20-series |
| 8.6 | Ampere (consumer) | RTX 30-series |
| 8.9 | Ada | RTX 40-series |

**Rationale**: this is the spread that covers essentially every GPU a home ZoneMinder user is
likely to own, at four architectures rather than seven. PTX for the newest lets a Blackwell or
later card JIT-compile at first load rather than failing outright — the "too new" edge case in the
spec — at the cost of a one-off delay that must be documented so it is not mistaken for a hang.

CUDA 12.4 supports 6.1 through 9.0; Pascal's removal comes with CUDA 13, which is the event that
will eventually force FR-017's Pascal clause to be revisited as a breaking change.

**Alternatives considered**: adding 7.0 (Volta) and 9.0 (Hopper) — datacentre parts that no
plausible user of this image owns, at roughly 50% more compile time against a 6-hour ceiling.
Dropping to 6.1 alone would halve the build and violate Principle V.

---

## R4. Installing pyzm, and the truth about the `opencv-python` hazard

**Decision**: install `pyzm[serve]` only, from a pinned git tag, and additionally install upstream's
own `opencv-python` shim as a defence in depth.

Read from `setup.py` at `v2.5.1`:

| Extra | Contents |
|---|---|
| base | `requests`, `pydantic>=2`, `dateparser`, `mysql-connector-python`, `python-dotenv` |
| `[ml]` | `numpy`, `Pillow`, `onnx>=1.12`, `Shapely`, `portalocker` |
| `[serve]` | `[ml]` + `fastapi>=0.100`, `uvicorn>=0.20`, `python-multipart`, `PyJWT`, `PyYAML` |
| `[train]` | `[ml]` + **`ultralytics>=8.3`**, `streamlit`, … |
| `[full]` | everything, including `ultralytics` |

**`ultralytics` appears only in `[train]` and `[full]`.** Nothing in `[serve]` declares
`opencv-python`, and nothing in it declares Ultralytics. The hazard recorded in the consuming
project's contract is therefore avoidable rather than merely survivable: install `[serve]`, never
`[full]`, and no dependency resolver is ever in a position to fetch a CPU-only OpenCV wheel over
the source build.

Note that `[serve]` declares no OpenCV dependency at all — `pyzm.ml` simply expects `import cv2` to
work. That is precisely what a source build provides, and it is why the install order still matters:
`cv2` must be importable before anything else is resolved.

**The shim, and why to keep it anyway.** Upstream's `scripts/setup_venv.sh:125-155` writes a fake
`opencv_python-<version>.dist-info` (METADATA, RECORD, WHEEL, top_level.txt) into site-packages when
a source-built `cv2` is importable, so pip believes `opencv-python` is already satisfied. We adopt
the same technique. It is redundant today given the extras above, and that is the point: it makes
the invariant hold even if a future dependency, or a future maintainer reaching for `[full]`,
introduces a package that wants the wheel. FR-009 requires the mechanism be commented in place with
the failure it prevents; citing upstream's own script in that comment gives the next reader
somewhere to go.

**Alternatives considered**: a pip constraints file pinning `opencv-python` to an impossible
version — fails loudly at install time rather than resolving correctly, and produces a confusing
error. Post-install `pip uninstall opencv-python` — races with layer caching and only helps if
someone remembers to re-run it.

---

## R5. Models: acquisition, export, and layout

**Decision**: ship all three model families baked in (FR-013, Principle VI). Obtain YOLO11 as
Ultralytics `.pt` weights and export to ONNX in an isolated build stage; obtain YOLOv4 as Darknet
weights; vendor the small text files into this repository.

**No pre-exported ONNX exists upstream.** Both `ultralytics/assets` (release `v8.4.0`, which
carries `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`) and the `Ultralytics/YOLO11` Hugging Face
repository publish `.pt` only. Exporting ourselves is the only way to satisfy FR-013 without asking
the operator to supply a model, which Principle VI forbids.

**Export stage design**: a throwaway stage on a plain slim Python base — *not* the CUDA base —
installing `ultralytics` and **CPU-only** PyTorch (`--index-url
https://download.pytorch.org/whl/cpu`), which avoids pulling several gigabytes of CUDA wheels for
an export that needs no GPU. It runs `model.export(format="onnx", imgsz=640, dynamic=False,
simplify=True, opset=17)`, and the final stage `COPY --from`s only the resulting `.onnx`. This is
also where the Ultralytics/`opencv-python` collision is contained: it happens in a stage whose
site-packages never reach the shipped image.

`imgsz=640` matches `yolo_onnx.py:25` (`_DEFAULT_DIM = 640`); exporting at another size and letting
the backend assume 640 would silently degrade accuracy.

**Leave `nms=False` (the default).** `yolo_onnx.py:90-147` detects end-to-end (NMS-baked) ONNX and
carries a pre-NMS fallback path for when "OpenCV produces garbled output" — a code path whose
existence is a warning. The plain export takes the well-trodden branch.

**Labels need no file.** `populate_class_labels()` (`yolo_onnx.py:72-88`) reads class names from
ONNX metadata via the `onnx` package when no labels file is configured, and Ultralytics embeds
`names` on export. A `coco.names` alongside the `.onnx` is therefore unnecessary. YOLOv4 Darknet
does need one, plus its `.cfg`.

**Pinning (Principle II, FR-014)**: pin the *inputs* — the `.pt` release asset by URL and SHA-256,
the `ultralytics` and `torch` versions, and the Darknet weights by SHA-256. The exported `.onnx` is
not bit-reproducible across toolchain versions, so its checksum is recorded as an observation, not
asserted as a gate. This is an honest limit of exporting rather than downloading, and it is the
reason the input pins matter more here than usual.

**Vendor the text files.** `yolov4.cfg` and `coco.names` are currently fetched from
`AlexeyAB/darknet@master` in most guides — a floating ref, which Principle II forbids. Both are
small text files; committing them to this repository removes two network dependencies from the
build and makes them reviewable in git. Only `yolov4.weights` (~250 MB, verified reachable at the
`darknet_yolo_v3_optimal` release tag) is fetched.

**On-disk layout**, matching `--base-path` and upstream's discovery rules (`serve.rst`):

```
/var/lib/zmeventnotification/models/
├── ultralytics/
│   ├── yolo11m.onnx      # primary — loaded by default
│   └── yolo11s.onnx      # present, not loaded unless requested
└── yolov4/
    ├── yolov4.weights    # legacy fallback — present, not loaded
    ├── yolov4.cfg
    └── coco.names
```

Discovery is by file-stem match in any subdirectory of the base path, so `--models yolo11m` resolves
to `ultralytics/yolo11m.onnx` and registers as `type: object`, `framework: opencv`.

**Actual SHA-256 values are not recorded in this plan.** They are captured during implementation and
committed to a checksums file; inventing them here would be worse than leaving them to a task.

---

## R6. The silent-fallback problem (most important finding)

**Finding**: upstream cannot detect an OpenCV built without CUDA, so a CUDA-less image logs nothing
unusual and serves CPU inference while reporting `processor:gpu`.

`YoloBase._setup_gpu()` (`pyzm/ml/backends/yolo.py:204-231`) does this:

1. If `processor == "gpu"` and `cv2.__version__ < 4.2.0`, log *"OpenCV %s does not support CUDA for
   DNNs (need 4.2+)"* and downgrade to CPU. **This is a version check, not a capability check.** A
   CUDA-less OpenCV 4.12 sails through it.
2. Otherwise call `setPreferableBackend(DNN_BACKEND_CUDA)` / `setPreferableTarget(DNN_TARGET_CUDA)`
   inside a `try`. On a CUDA-less build these calls **do not raise** — OpenCV accepts the request and
   quietly falls back to the CPU backend at `forward()` time. The `except` never fires.
3. The *"requested processor=%s but fell back to %s"* message at `yolo.py:79-85` is emitted only
   when `self.processor` was actually reassigned by one of the paths above.

So in the precise failure case that cost the predecessor stack a year on CPU, all three log
signatures are absent. The consuming project's contract verification —

```bash
docker logs pyzm-test 2>&1 | grep -iE "fell back|does not support CUDA"   # must be empty
```

— passes. It is a useful check for *other* failures (a genuine runtime CUDA error at
`yolo.py:110-131` does log and does fall back), but on its own it is exactly what Constitution
Principle I calls "a check that could pass while the GPU sits idle".

**Consequence — the verification ladder.** Three independent checks, none of which can be satisfied
by the others:

| Layer | Check | Catches | Where |
|---|---|---|---|
| Build | `cv2.getBuildInformation()` reports NVIDIA CUDA: YES, and the `cv2.cuda` bindings exist | a CUDA-less compile, a shadowing wheel, a contrib-less build | Dockerfile `RUN`, fails the build (FR-008/009) |
| Publish | same assertion, run against the image **pulled back from the registry** | bad cache hit, wrong tag, registry mixup | CI job after push (FR-010) |
| Runtime | `cv2.cuda.getCudaEnabledDeviceCount() > 0`, plus observed `nvidia-smi` utilisation during a real inference | no GPU, no driver, no `--gpus`, arch not built for | operator, documented in README (FR-012) |

The build and publish layers run happily without a GPU, which is what makes them usable in CI
(FR-027). Only the third proves execution, and only a human with hardware can run it.

**Alternatives considered**: parsing pyzm's logs at container start. Rejected — this finding is
precisely that those logs are silent in the failure case.

---

## R7. Entrypoint GPU preflight

**Decision**: the entrypoint runs a preflight check before starting the server. When the processor
is `gpu` (our default), it asserts `cv2.cuda.getCudaEnabledDeviceCount() > 0` and, if that fails,
prints a specific diagnostic and exits non-zero. `PYZM_SERVE_ALLOW_CPU=1` downgrades the failure to
a loud warning for operators who genuinely want CPU.

**Rationale**: FR-004 requires an explicit, visible failure when GPU inference is requested and
cannot be provided. R6 shows upstream will not provide one, and Principle III forbids us from
adding endpoints or altering their semantics. A preflight in the entrypoint is neither: it is the
container declining to start in a configuration the operator did not ask for. A container that
exits with *"GPU inference was requested but no CUDA device is visible — did you pass `--gpus
all`?"* is strictly better than one that runs at a fifth of the speed and says nothing.

The escape hatch matters for Principle V: someone without a GPU may still want to try the image,
and refusing outright with no override would be an assumption about the user's hardware.

**Alternatives considered**: defaulting `--processor cpu` (upstream's default) and making GPU
opt-in. Rejected — this is a GPU image; a user who gets CPU by default gets the predecessor's bug
back as a feature.

---

## R8. Build time and caching

**Constraint**: GitHub-hosted runners cap a job at **6 hours**, and larger runners require a Team or
Enterprise plan, which a personal account does not have. Standard runners are free for public
repositories. The OpenCV CUDA compile across four architectures is the only step at risk.

**Decisions**:

1. **`BUILD_LIST` (R2)** is the main lever — it is the difference between compiling ten modules and
   sixty, multiplied by four architectures.
2. **Registry-backed BuildKit cache**: `cache-from`/`cache-to` of `type=registry,mode=max` against a
   dedicated `:buildcache` tag in GHCR. Preferred over `type=gha`, whose 10 GB repository budget an
   OpenCV build blows through. Compliance note: `:buildcache` is mutable by design, and is neither a
   release tag nor a runnable image, so it does not conflict with FR-024 — but it must be named so
   nobody mistakes it for one.
3. **Layer ordering (FR-019)**: base → system packages → OpenCV compile → model export → pyzm
   install → application files. Only the last two invalidate on ordinary changes.
4. **`ccache`** inside the compile stage, persisted through the same registry cache.
5. **An explicit `timeout-minutes`** below the hard cap, so a runaway build fails as a timeout with a
   clear cause rather than being killed at six hours.

**Escalation order** if the cold build still cannot fit. Each step costs more than the one above it,
and the last two are decisions for the operator, not an agent:

1. **Trim `BUILD_LIST` further.** Cheapest, invisible to users.
2. **Drop an architecture**, documented — it narrows FR-017. Prefer dropping the *newest* SASS target
   and letting PTX cover it: JIT compiles forward, so SASS at 6.1/7.5/8.6 with PTX at 8.6 still runs
   on Ada and Blackwell, at a one-off first-load delay. This reduces the per-architecture multiplier
   with no change to how anyone deploys.
3. **Publish one image per compute capability** (`v<semver>-sm61`, `-sm75`, …) from a job matrix, one
   architecture per job. The 6-hour limit is per *job*, so this takes the critical path from
   `A + 4K` to `A + K`, where `A` is the architecture-independent compile and `K` the per-architecture
   CUDA kernel compile — only `.cu` sources multiply with `CUDA_ARCH_BIN`. Total CPU-hours rise,
   which is free for a public repository. Ranked above option 4 because it keeps CI automated, and
   below option 2 because it is the first step that changes what a *user* must know:
   - **Nothing auto-selects the right image.** Docker manifest lists key on CPU architecture, not GPU
     compute capability, so every user must map GPU → capability → tag by hand. Getting it wrong
     fails at first inference on their own hardware with "no kernel image is available for execution
     on the device" — loud, but late, and after they have followed the README. That is a direct cost
     to SC-005 and Principle V.
   - It multiplies the verification surface: the gate-bites check, the pull-back re-assertion and the
     release job all run per image.
   - Images get materially smaller, which helps SC-008.
   - It rewrites the tag namespace in `contracts/container-interface.md` §1 and §9, the workflow
     matrix, and the README's pinning and GPU-generation sections. Taking it late means redoing that
     work — not breaking a promise: no real release exists until the spec is complete, so FR-028 has
     nothing to bite on while this is still open.
4. **Move the OpenCV stage to a separately published, digest-pinned base image built manually.** Keeps
   CI green at the cost of one manual step outside the pipeline.

Note that the benefit of option 3 depends entirely on the `K/A` ratio, which is unmeasured until the
first cold build is timed. If `A` dominates, it quadruples the CI surface to shave little from the
critical path. Measure before restructuring.

---

## R9. CI/CD

**Decision**: two workflows, mirroring `docker-zm-mlapi`'s shape with Docker Hub removed and
verification added.

| | `build.yml` | `release.yml` |
|---|---|---|
| Trigger | push to `main`, `workflow_dispatch` | push of a tag |
| Tag published | `main-build<run_id>-<sha>` | `<git tag>` (immutable) and `latest` |
| Verification | build-time assertion only | build-time **plus** pull-back re-verification |
| Release | none | GitHub release created |

- `permissions: {contents: write, packages: write}`; `GITHUB_TOKEN` authenticates to GHCR, so
  **no repository secret is required at all** now that Docker Hub is gone. This is a real
  simplification worth calling out in the README: the sibling repository's `DOCKERHUB_TOKEN` setup
  step disappears.
- `docker/build-push-action@v6` with `sbom: true` and `provenance: mode=max` (FR-025), plus the four
  OCI labels the sibling repository already sets.
- `platforms: linux/amd64` only (FR-020). Do not enable QEMU multi-arch: an emulated CUDA compile
  would not finish inside any timeout.
- The pull-back verification is a **separate job** that `needs:` the build, so it runs against the
  registry rather than against build-local state. That separation is the entire point of FR-010.
- `latest` is moved on release but never recommended for deployment (FR-024); the README points at
  immutable tags only.
- The build-time CUDA assertion lives in the Dockerfile, not the workflow, so `docker build` locally
  gets the same gate CI does.

**Alternatives considered**: a single workflow keyed on ref type — fewer files, but conflates
"publish something to try" with "make a permanent promise", and those deserve visibly different
code paths.

---

## R10. Healthcheck

**Decision**: `HEALTHCHECK` invoking a Python one-liner (`urllib.request`) against
`http://127.0.0.1:${PYZM_SERVE_PORT}/health`, with a start period long enough to cover model load.

`GET /health` returns `{"status": "ok", "models_loaded": true}` (`serve.rst`, API reference).
`curl` is not in the runtime base image and adding it for a healthcheck is a wasteful dependency
when Python is guaranteed present. The start period must accommodate first-load PTX JIT on a
too-new GPU (R3), which can take tens of seconds — a start period that is too short turns that edge
case into a crash loop.

**Note on `--models all`**: with `all`, weights load lazily on first request, so `models_loaded`
would report false until traffic arrives. Our default names `yolo11m` explicitly, which loads
eagerly and makes the healthcheck meaningful. Operators who switch to `all` should expect a
healthcheck that reflects lazy loading.

---

## R11. Runtime configuration surface

**Decision**: environment variables with defaults, consumed by the entrypoint and turned into
upstream CLI flags. No configuration file (Principle III explicitly forbids reintroducing one, and
upstream's own `--config` YAML is left available but undocumented by us).

| Variable | Default | Maps to |
|---|---|---|
| `PYZM_SERVE_MODELS` | `yolo11m` | `--models` |
| `PYZM_SERVE_PROCESSOR` | `gpu` | `--processor` |
| `PYZM_SERVE_PORT` | `5000` | `--port` |
| `PYZM_SERVE_HOST` | `0.0.0.0` | `--host` |
| `PYZM_SERVE_BASE_PATH` | `/var/lib/zmeventnotification/models` | `--base-path` |
| `PYZM_SERVE_WORKERS` | `1` | `--workers` |
| `PYZM_SERVE_DEBUG` | unset | `--debug` |
| `PYZM_SERVE_AUTH` | unset (off) | `--auth`, with user/password/secret vars |
| `PYZM_SERVE_ALLOW_CPU` | unset | R7 escape hatch |

Defaults differ from upstream's in two places, both deliberate: `--processor` (upstream `cpu`, ours
`gpu`) and `--models` (upstream `yolo11s`, ours `yolo11m`). Both must be stated in the README,
because a user coming from upstream's documentation will otherwise be surprised.

Any additional argument passed to `docker run` after the image name is appended to the server
command line, so operators are never boxed in by this table.

**Auth defaults (FR-007)**: off. Upstream's `--token-secret` defaults to the literal `change-me`,
which is safe only because auth is disabled; enabling auth without setting a secret must be refused
by the entrypoint rather than silently signing tokens with a published default.

---

## R12. Image size and layout

Expected shipped size is roughly **2.5–3 GB** compressed: 2.07 GB of CUDA/cuDNN runtime, ~200–300 MB
of OpenCV libraries and Python bindings, ~310 MB of models (`yolo11m.onnx` ~40 MB, `yolo11s.onnx`
~19 MB, `yolov4.weights` ~250 MB), and a small amount of Python dependencies.

This is large, and it is the direct price of Principle VI plus a CUDA runtime. FR-018 and SC-008
require that no compiler toolchain contributes to it, and the README must state the number so a user
knows before pulling (SC-008). The exact figure is measured during implementation, not guessed here.

**Considered and rejected**: dropping `yolov4.weights` to save 250 MB. It is the fallback known to
work with OpenCV DNN's CUDA backend, and the Constitution names it explicitly under Models.
Fetching it at run time would save the space and violate Principle VI.

---

## Sources

- [pyzmNg `serve.rst` @ v2.5.1](https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/docs/guide/serve.rst) — CLI flags, defaults, discovery rules, endpoints
- [pyzmNg `setup.py` @ v2.5.1](https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/setup.py) — extras (R4)
- [pyzmNg `yolo.py` @ v2.5.1](https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/pyzm/ml/backends/yolo.py) — `_setup_gpu`, fallback logging (R6)
- [pyzmNg `yolo_onnx.py` @ v2.5.1](https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/pyzm/ml/backends/yolo_onnx.py) — `_DEFAULT_DIM`, label metadata, end2end handling (R5)
- [pyzmNg `setup_venv.sh` @ v2.5.1](https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/scripts/setup_venv.sh) — the `opencv-python` shim (R4)
- [pyzmNg installation guide](https://pyzmng.readthedocs.io/en/latest/guide/installation.html)
- [nvidia/cuda tags on Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags) — digests (R1)
- [opencv/opencv releases](https://github.com/opencv/opencv/releases) — version landscape (R2)
- [opencv#24983 — DNN fails to compile against cuDNN 9.0](https://github.com/opencv/opencv/issues/24983)
- [OpenCV forum — 4.12.0 / CUDA 12.9 cub errors](https://forum.opencv.org/t/persistent-cuda-cudnn-build-errors-4-12-0-cuda-12-9-vs2022/23451)
- [Ultralytics/YOLO11 on Hugging Face](https://huggingface.co/Ultralytics/YOLO11) — `.pt` only (R5)
- [ultralytics/assets releases](https://github.com/ultralytics/assets/releases) — `yolo11{n,s,m}.pt` (R5)
- [Ultralytics export documentation](https://docs.ultralytics.com/modes/export) — ONNX export parameters (R5)
- [GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) — 6-hour job cap, larger-runner plan requirements (R8)
