---

description: "Task list for 001-pyzm-serve-gpu-image"
---

# Tasks: GPU-Accelerated pyzm.serve Inference Gateway Image

**Input**: Design documents from `/specs/001-pyzm-serve-gpu-image/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: No unit-test suite. The spec does not request one and there is no application logic to
unit-test — Principle III caps our own code at an entrypoint and two verification scripts.
Verification is instead **executable and layered**, and its tasks are first-class implementation
tasks below, mapped to scenarios in [quickstart.md](./quickstart.md). A verification task is not
optional polish here; FR-008 through FR-012 are requirements.

**Organization**: Grouped by user story so each is independently implementable and testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1–US4)
- Exact file paths are given in every task

## Path Conventions

Flat repository root, matching `docker-zm-mlapi`. The deliverable is one image, so there is no
`src/` or `tests/` tree. Paths below are repository-relative from `/home/jantman/GIT/docker-pyzm-serve`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Repository scaffolding and the two vendored files that must not be fetched from a
floating ref.

- [X] T001 Create `.dockerignore` excluding, at minimum: `.git/`, `specs/`, `.specify/`, `.claude/`, `**/*.md`, `LICENSE`, and — **the same downloaded model binaries T002 keeps out of git** — `**/*.pt`, `**/*.onnx`, `**/*.weights`, `sample.*`. Two distinct reasons, both load-bearing: excluding the documentation tree is what stops a README edit invalidating the multi-hour OpenCV layer (FR-019), and excluding the model binaries stops T005's local downloads — several hundred MB that the build fetches for itself in T013/T014 — being uploaded into the build context on every `docker build`. Use `**/*.md` rather than `*.md`: the bare form matches only the context root, so `models/README.md` would still be sent. It must **not** exclude `scripts/` or `models/` — T011 and T018 `COPY` the assertion script into the image, and T003/T004's vendored `yolov4.cfg` and `coco.names` are build inputs. Cheap acceptance signal: `docker build` reports a transfer context in the low single-digit MB, not hundreds
- [X] T002 Create `.gitignore` for local build scratch — `*.onnx`, `*.pt`, `*.weights`, `sample.jpg`, `sample.png` — so the large model inputs T005 downloads to compute checksums are never committed. Keep this list and T001's in step: the same artifacts must be invisible to both git and the build context, and a file that appears in only one of the two lists is the failure mode this pairing exists to prevent. What *is* committed is `models/checksums.sha256` — the checksums are the pinned record (PI-3); the binaries they describe are fetched and verified at build time (FR-014)
- [X] T003 [P] Vendor `models/yolov4.cfg` from AlexeyAB/darknet, copied from a specific commit and with that commit SHA recorded in a comment at the top of the file — the upstream path is a branch ref, which Principle II forbids (research R5)
- [X] T004 [P] Vendor `models/coco.names` (80 COCO class names) the same way, for the YOLOv4 Darknet model only — YOLO11 ONNX reads its labels from embedded metadata (`yolo_onnx.py:72-88`) and needs no labels file

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Resolve and record every pinned input, and write the CUDA assertion that the build,
CI, and every later verification step all execute.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete. T008 in particular is
the single gate that Principle I hangs on, and it is shared by three different callers.

- [X] T005 Download `yolo11m.pt` and `yolo11s.pt` (ultralytics/assets release `v8.4.0`) and `yolov4.weights` (AlexeyAB/darknet release `darknet_yolo_v3_optimal`), compute SHA-256 for each, and record them in `models/checksums.sha256` in `sha256sum` format. Do **not** commit the downloaded binaries. These are the real values the plan deliberately did not invent (research R5, FR-014)
- [X] T006 Resolve the three upstream git tags to full commit SHAs for use as ARG defaults in T007: OpenCV `4.12.0`, `opencv_contrib` `4.12.0`, pyzmNg `v2.5.1`. Tags are mutable; Principle II requires full SHAs
- [X] T007 Create `Dockerfile` containing only the `ARG` declarations and their pinned defaults — `CUDA_DEVEL_IMAGE` and `CUDA_RUNTIME_IMAGE` (digests in [data-model.md](./data-model.md)), `OPENCV_REF`, `OPENCV_CONTRIB_REF`, `PYZM_REF` (from T006), `YOLO11_ASSETS_REF`, `YOLOV4_WEIGHTS_URL`, `ULTRALYTICS_VERSION`, `TORCH_VERSION`, `CUDA_ARCH_BIN="6.1;7.5;8.6;8.9"`, `CUDA_ARCH_PTX="8.9"`, `OPENCV_CUDA=ON`. Stages are added in Phase 3. `OPENCV_CUDA` exists so the gate can be proven to bite (T024). `CUDA_ARCH_PTX` is what FR-017's forward-compatibility clause actually requires — without it the image ships SASS for four architectures and **nothing** for Blackwell and later, while [contracts/container-interface.md](./contracts/container-interface.md) §9 and T045 both promise those cards work. `YOLO11_ASSETS_REF` pins the `ultralytics/assets` release the two `.pt` files come from, per [data-model.md](./data-model.md)
- [X] T008 [P] Write `scripts/assert-cuda-build.py`: import `cv2`, assert `getBuildInformation()` reports NVIDIA CUDA as `YES`, assert the `cv2.cuda` namespace is present (proving the contrib bindings were built), print the OpenCV version and the CUDA build lines, and exit non-zero with an explicit message on failure. It MUST NOT require a GPU — it tests what was compiled, not what is available (FR-008, FR-027). This one file is executed by the Dockerfile, by the CI pull-back job, and by quickstart Scenario 5a; keeping it a file rather than an inline string is what stops those three drifting apart

**Checkpoint**: Every external input is pinned and checksummed, and the assertion that guards them all exists.

---

## Phase 3: User Story 1 - Run object detection on the GPU (Priority: P1) 🎯 MVP

**Goal**: A container image that serves GPU object detection, refuses to start silently on CPU, and
cannot be built without compiled-in CUDA support.

**Independent Test**: Build locally, run with `--gpus all` on a GPU host, post a frame to `/infer`,
get detections back while `nvidia-smi` shows utilisation and no CPU-fallback message appears.
Quickstart Scenarios 1–5.

### Image build

- [X] T009 [US1] Add the `opencv-build` stage to `Dockerfile` on `CUDA_DEVEL_IMAGE`: install build dependencies (`build-essential`, `cmake`, `git`, `ccache`, `python3-dev`, `python3-numpy`, `pkg-config`), clone OpenCV and `opencv_contrib` at the pinned SHAs, and configure with the full CMake flag set from [research.md](./research.md) R2 — `WITH_CUDA=${OPENCV_CUDA}`, `WITH_CUDNN=ON`, `OPENCV_DNN_CUDA=ON`, `WITH_CUBLAS=ON`, `CUDA_ARCH_BIN=${CUDA_ARCH_BIN}`, **`CUDA_ARCH_PTX=${CUDA_ARCH_PTX}`**, `OPENCV_EXTRA_MODULES_PATH`, the trimmed `BUILD_LIST`, and all `BUILD_TESTS`/`BUILD_PERF_TESTS`/`BUILD_EXAMPLES`/`BUILD_DOCS`/`BUILD_JAVA` off. `CUDA_ARCH_PTX` is not optional decoration: it emits the intermediate code a too-new card JITs at first load (research R3, FR-017), and it is the only line here that anything outside the build depends on by name — T023's healthcheck start period and T045's README section are both written assuming it is present
- [X] T010 [US1] Add the compile and install steps to the `opencv-build` stage, into a staging prefix that the runtime stage can `COPY --from`. `opencv_contrib` is **not** optional: `OPENCV_DNN_CUDA=ON` needs `cudev`, and `cudaarithm` is what puts `cv2.cuda` into the Python bindings, making it load-bearing for verification rather than incidental
- [X] T011 [US1] Add a `RUN` to the end of the `opencv-build` stage that executes `scripts/assert-cuda-build.py` against the freshly built OpenCV, failing the build if CUDA is absent (FR-008)
- [X] T012 [P] [US1] Write `scripts/export-onnx.py`: load a `.pt` with Ultralytics and export with `format="onnx"`, `imgsz=640`, `dynamic=False`, `simplify=True`, `opset=17`. `imgsz=640` matches `yolo_onnx.py:25`'s `_DEFAULT_DIM` — exporting at another size silently mis-scales every detection. Leave `nms=False` (the default) to stay on upstream's well-trodden parsing path rather than its end-to-end fallback branch (research R5)
- [X] T013 [US1] Add the `model-export` stage to `Dockerfile` on a **slim Python base, not the CUDA base**: install pinned `ultralytics` and **CPU-only** PyTorch (`--index-url https://download.pytorch.org/whl/cpu`), fetch the two `.pt` files, verify them against `models/checksums.sha256`, and run `scripts/export-onnx.py`. CPU-only torch avoids pulling gigabytes of CUDA wheels for an export needing no GPU, and this throwaway stage is where the Ultralytics/`opencv-python` collision is contained (research R4)
- [X] T014 [US1] Add a `models` stage (or extend `model-export`) that downloads `yolov4.weights` from the pinned release URL and verifies it against `models/checksums.sha256`, failing the build on mismatch (FR-014)
- [X] T015 [US1] Add the runtime stage to `Dockerfile` on `CUDA_RUNTIME_IMAGE`: install the minimal Python runtime, `COPY --from` only the OpenCV install tree and the exported/downloaded model files, and assemble `/var/lib/zmeventnotification/models/` with `ultralytics/{yolo11m,yolo11s}.onnx` and `yolov4/{yolov4.weights,yolov4.cfg,coco.names}` per [data-model.md](./data-model.md). No compiler toolchain may reach this stage (FR-018)
- [X] T016 [US1] Install upstream's `opencv-python` dist-info shim in the runtime stage before installing pyzm, following `scripts/setup_venv.sh:125-155`, and **comment it with the failure it prevents and a link to that upstream script**. FR-009 requires the explanation in place: this looks arbitrary to anyone who has not been bitten, and a future tidy-up that deletes it restores a year-long silent bug
- [X] T017 [US1] Install `pyzm[serve]` from the pinned SHA in the runtime stage. **`[serve]`, never `[full]`** — `[full]` and `[train]` pull Ultralytics and therefore `opencv-python`, which would shadow the source build; `[serve]` declares neither (research R4). Add a comment saying so, because `[full]` looks like the obviously-safer choice
- [X] T018 [US1] Add the final `RUN scripts/assert-cuda-build.py` to the runtime stage, asserting against the OpenCV that the shipped image will actually load — after the shim and after pyzm, so it catches anything those steps could have disturbed (FR-009)
- [X] T019 [US1] Add runtime-stage metadata to `Dockerfile`: `EXPOSE 5000`, a non-root user, `ENV` defaults for the whole variable table in [contracts/container-interface.md](./contracts/container-interface.md) §5, and `ENTRYPOINT ["/opt/entrypoint.sh"]`. Declare no `VOLUME` — the image needs no writable storage (FR-033)

### Entrypoint

- [X] T020 [US1] Write `entrypoint.sh`: map every `PYZM_SERVE_*` variable to its upstream CLI flag per [contracts/container-interface.md](./contracts/container-interface.md) §5, append any trailing `docker run` arguments verbatim, and `exec` `python -m pyzm.serve` so it becomes PID 1 and receives `SIGTERM` directly
- [X] T021 [US1] Add the GPU preflight to `entrypoint.sh`: when the processor is `gpu`, assert `cv2.cuda.getCudaEnabledDeviceCount() > 0` and on failure exit `78` with a message naming `--gpus all` as the fix; `PYZM_SERVE_ALLOW_CPU=1` downgrades it to a loud warning. This exists because upstream cannot detect a CUDA-less OpenCV — `setPreferableBackend(DNN_BACKEND_CUDA)` succeeds on a CPU-only build and nothing is logged (research R6), so without this FR-004 is unsatisfiable
- [X] T022 [US1] Add the auth guard to `entrypoint.sh`: with auth enabled and **either** `PYZM_SERVE_TOKEN_SECRET` **or** `PYZM_SERVE_AUTH_PASSWORD` unset, exit `78` naming the missing variable (FR-007). The token-secret half refuses to sign with upstream's published `change-me` default; the password half exists because [contracts/container-interface.md](./contracts/container-interface.md) §5 marks `PYZM_SERVE_AUTH_PASSWORD` required-when-auth-on and gives it no default, so auth-on-without-password would otherwise start with whatever upstream falls back to — an authenticated endpoint whose credentials are in someone else's source tree is worse than an unauthenticated one, because the operator believes it is protected
- [X] T023 [US1] Add `HEALTHCHECK` to `Dockerfile` hitting `http://127.0.0.1:${PYZM_SERVE_PORT}/health` via `urllib.request` (no `curl` in the runtime image) — the **configured** port, not a hardcoded `5000`, or every operator who moves the port gets a container that is permanently unhealthy while serving perfectly (research R10) — with a start period long enough to cover model load **including first-load PTX JIT on a GPU newer than any built for** — too short a period turns that supported edge case into a crash loop (research R10)

### Verification — User Story 1

- [X] T024 [US1] Prove the CUDA gate bites: run `docker build --build-arg OPENCV_CUDA=OFF .` and confirm it **fails** at the assertion with nothing tagged (quickstart Scenario 1, SC-003). A gate nobody has watched fail is not known to work — this is the check whose absence cost the predecessor a year
- [X] T024a [US1] Prove the **checksum** gate bites too: corrupt one line of `models/checksums.sha256`, confirm the build fails at the verification step in T013 or T014 with a message naming the offending artifact, then restore the file (FR-014, quickstart Scenario 1). T024 exists because a gate nobody has watched fail is not known to work; that reasoning applies unchanged here. A model that changes silently produces detection differences indistinguishable from a code regression, and this is the only thing standing between a user and that
- [X] T025 [US1] Run the self-containment checks from quickstart Scenario 2: model tree present with `--network none`, no `gcc`/`g++`/`cmake`/`nvcc` in the image, and identical model checksums across two containers from the same tag (FR-018, FR-033, FR-034, SC-014)
- [X] T025a [US1] Run quickstart Scenario 2c — the **offline serve** check: start the container with `--network none` and no volumes, then from *inside* it (`docker exec`, since `--network none` makes published ports impossible) poll `/health` until healthy and post a frame to `/infer`, confirming detections come back. This is SC-013 and it is the test Constitution Principle VI names for itself — "the image MUST start, become healthy, and answer an inference request with no network access and no volumes mounted. This is the test of this principle, and it is cheap enough to run every time." Listing model files offline (T025) is not that test. On a GPU-less machine, pair with `PYZM_SERVE_PROCESSOR=cpu PYZM_SERVE_ALLOW_CPU=1`; the claim under test is self-containment, not GPU execution (FR-013, FR-033, SC-013)
- [X] T026 [US1] Run the loud-failure checks from quickstart Scenario 3: no-GPU start exits `78`, `PYZM_SERVE_ALLOW_CPU=1` starts with a warning, and auth-without-secret exits `78` (FR-004, FR-007)
- [X] T026a [US1] Verify the HTTP surface is exactly upstream's: fetch `GET /openapi.json` from a running container, extract the route list, and confirm it matches the four documented upstream endpoints — health, model listing, inference, login — with nothing added, removed, or renamed (FR-002, SC-010, Principle III). Needs no GPU: run it under `PYZM_SERVE_PROCESSOR=cpu PYZM_SERVE_ALLOW_CPU=1`. A negative requirement with no check is a hope, and this one is the entire basis of SC-010's promise that an existing client works unmodified
- [X] T027 [US1] On a GPU host, run quickstart Scenarios 4 and 5: health, `/models` showing `yolo11m` loaded, a real `/infer` returning detections, p95 latency at or under 300 ms, non-zero `cv2.cuda.getCudaEnabledDeviceCount()`, and observed `nvidia-smi` utilisation across 100 sustained requests (SC-001, SC-002)
- [X] T028 [US1] Write `docker-compose.yml`: a worked example reserving GPU access, publishing port 5000, with no volumes required — the baseline Principle V asks for and FR-031 requires

**Checkpoint**: The image exists, works on a GPU, and cannot be built or started in the silent-CPU state. This is the MVP.

---

## Phase 4: User Story 2 - Cut an immutable, verified release (Priority: P2)

**Goal**: A git tag produces an immutable, independently re-verified image in GHCR and a GitHub
release, with no repository secret required.

**Independent Test**: Push a throwaway pre-release tag; confirm the tagged image appears in GHCR,
the pull-back verification job ran and passed, a GitHub release was created, and nothing was pushed
to Docker Hub. Quickstart Scenario 7.

**Depends on**: Phase 3 — there must be a Dockerfile to build. Stated plainly rather than
pretending independence.

- [X] T029 [P] [US2] Write `scripts/verify-published.sh`: take an image reference, `docker pull` it from GHCR, and run `scripts/assert-cuda-build.py` against that pulled image, exiting non-zero on failure. The script is **not baked into the runtime image** (T015 copies only the OpenCV tree and the models), so mount it in — `docker run --rm -v "$PWD/scripts:/verify:ro" "$IMAGE" python3 /verify/assert-cuda-build.py` — rather than assuming a path inside the image. It must operate on the **pulled** image; running against build-local state would defeat FR-010, whose entire purpose is catching a bad cache hit, a wrong tag, or a registry mixup
- [X] T030 [US2] Create `.github/workflows/release.yml` triggered on tag push, with `permissions: {contents: write, packages: write}`, GHCR login via `GITHUB_TOKEN`, and `docker/setup-buildx-action`. **No repository secret is needed** now that Docker Hub is gone — the sibling repo's `DOCKERHUB_TOKEN` setup step disappears entirely
- [X] T030a [US2] Add a **tag-immutability guard job** to `release.yml` that runs before the build and fails the workflow if the pushed tag already exists in GHCR (`docker buildx imagetools inspect ghcr.io/<owner>/docker-pyzm-serve:${{ github.ref_name }}` succeeding means it is already published). Make the build job `needs:` it. FR-024 and Constitution Principle IV say a published tag MUST NEVER be overwritten, moved, or deleted — but nothing enforces that on its own: GHCR happily accepts a second push to the same tag, and `docker/build-push-action` has no opinion about it. Without this guard, re-running `release.yml` for an existing tag silently moves it, and T036's digest-unchanged check fails, because this build is not bit-reproducible ([plan.md](./plan.md) records the ONNX export is not). Someone, somewhere, has pinned that tag
- [X] T031 [US2] Add the build-and-push job to `release.yml` using `docker/build-push-action@v6`: `platforms: linux/amd64` only, `sbom: true`, `provenance: mode=max`, the four OCI labels from [contracts/cicd-contract.md](./contracts/cicd-contract.md) §3, tags `<git tag>` and `latest`, and registry-backed cache (`type=registry,mode=max`, ref `:buildcache`). Do not enable QEMU multi-arch — an emulated CUDA compile finishes inside no timeout
- [X] T032 [US2] Add a **separate** verification job to `release.yml` with `needs:` the build job, invoking `scripts/verify-published.sh` against the published tag. Separation from the build job is the whole point of FR-010
- [X] T033 [US2] Add the GitHub release step to `release.yml`, gated on the verification job so a release is never presented as usable without it (FR-026). Use a maintained action — `actions/create-release` is archived
- [X] T034 [US2] Add a **Releasing** section at the bottom of `README.md` defining the release-note convention that FR-028 requires: releases are cut by pushing a git tag, the image tag equals the git tag, and the notes MUST explicitly call out any of the four breaking changes — a removed model, a changed default, a dropped GPU generation, or a new required setting — together with the version bump that signals it. State that a user must be able to judge from the tag alone whether an upgrade is safe, that a published tag is never moved or deleted, and that a release is not confirmed until it has been pulled and verified on real GPU hardware. Append-only: if Phase 6 has not run yet, this section can exist before the rest of the README is written
- [X] T035 [US2] Set an explicit `timeout-minutes` below the 6-hour hard cap on every job in `release.yml`, so a runaway OpenCV build fails as a legible timeout rather than being killed at the ceiling (research R8)
- [ ] T036 [US2] Verify with a throwaway tag per quickstart Scenario 7: correct tag published, verification job ran, release created, four OCI labels plus SBOM and provenance present, no Docker Hub push, and — re-running the workflow for that same tag — the guard job (T030a) fails the run before the build and the tag's digest is unchanged (FR-021 to FR-026, SC-004)

**Checkpoint**: Releases are immutable, verified, and reproducible by anyone who forks the repository.

---

## Phase 5: User Story 3 - Get fast feedback on every change (Priority: P3)

**Goal**: Every push to `main` builds and publishes under a tag that could never be mistaken for a
release, and a change that breaks the GPU path fails before publishing.

**Independent Test**: Push to `main`, confirm a `main-build<run_id>-<sha>` image appears; push a
change that breaks the CUDA build and confirm the pipeline fails rather than publishing.

**Depends on**: Phase 3. Note `build.yml` is very nearly a subset of `release.yml`, so implementing
this phase before Phase 4 is a legitimate reordering if you want the cheaper feedback loop first —
the spec's priority reflects value, not build order.

- [X] T037 [US3] Create `.github/workflows/build.yml` triggered on push to `main` and `workflow_dispatch`, with `permissions: {packages: write}` only, publishing `main-build${{ github.run_id }}-${{ github.sha }}` — a tag whose prefix and run ID make collision with a release tag impossible (FR-022, RT-1)
- [X] T038 [US3] Give `build.yml` the same labels, SBOM, provenance, registry cache and `timeout-minutes` as `release.yml`, so a `main` build is a faithful rehearsal of a release build and differs only in what it is called
- [ ] T039 [US3] Verify the failure path: push a commit that breaks the GPU build (for example forcing `OPENCV_CUDA=OFF`), confirm the pipeline fails and publishes nothing, then revert (FR-022, SC-003)
- [ ] T040 [US3] Verify `workflow_dispatch` triggers an identical build and publish with no code change (User Story 3, scenario 4)

**Checkpoint**: Mistakes surface on push instead of at release time, after a multi-hour build.

---

## Phase 6: User Story 4 - Adopt the image as a stranger (Priority: P4)

**Goal**: Someone who has never seen this repository reaches a working, GPU-verified gateway on
their own hardware from the README alone.

**Independent Test**: Hand the README to someone with a GPU host and no knowledge of the project;
they reach a verified gateway in under 30 minutes using only copy-pasteable commands, without asking
a question. Quickstart Scenario 8.

- [X] T041 [P] [US4] Write the `README.md` header: repostatus badge, one-line description, and the honest framing Principle V requires — personal project, best-effort, issues may not be addressed, PRs welcome but not guaranteed a review. Match `docker-zm-mlapi`'s tone; setting expectations is a feature (FR-030)
- [X] T042 [US4] Add prerequisites and quickstart to `README.md`: NVIDIA driver ≥ 550, NVIDIA Container Toolkit, the `docker run --gpus all` one-liner, and the `docker-compose.yml` walkthrough, ending at a working `/infer` call (FR-029)
- [X] T043 [US4] Add the GPU verification section to `README.md`, reproducing quickstart Scenario 5's three checks verbatim as copy-pasteable commands — **including the warning that an empty log-grep is not proof of GPU execution**. A user who cannot easily verify will assume it works, which is exactly how the predecessor's bug survived (FR-012, Principle I)
- [X] T044 [US4] Add the configuration reference to `README.md`: the full environment-variable table, an explicit callout of the two defaults that diverge from upstream (`--processor gpu` not `cpu`, `--models yolo11m` not `yolo11s`) so a user following upstream's docs is not surprised, and — FR-007's second half — a plain statement of **what authentication being off by default means**: the gateway accepts any request that can reach its port, which assumes a trusted network segment alongside its client, and `PYZM_SERVE_AUTH` with a password and token secret is how an operator whose situation differs turns it on. A default is only safe if the person relying on it knows what it assumes (FR-007, FR-029, RC-1)
- [X] T045 [US4] Add the capability reference to `README.md`: models shipped and how to request each, supported GPU generations with the too-new (PTX JIT, first-load delay) and too-old (unsupported) cases spelled out, how to supply your own model, how to pin a version, and that `latest` is not a deployment target (FR-024, FR-029, SC-011)
- [X] T046 [US4] Measure the published image's download size and state it in `README.md` so a user knows what they are pulling before they pull it (SC-008)
- [X] T047 [US4] Run the author-specific-values check from quickstart Scenario 8 across the **whole repository** — `grep -rIiE` over the tracked tree, excluding `.git/` and `specs/` — and additionally over the built image's environment and labels (`docker inspect`). FR-032 and SC-012 say "anywhere in the repository or image"; scanning only the four hand-picked files leaves `scripts/`, `.github/workflows/`, `models/` and the image's own `ENV` block unexamined, and a hostname baked into an `ENV` default is exactly the kind of thing that reaches a stranger's machine. Expect no matches (FR-032, SC-012)

**Checkpoint**: The project is honestly documented and usable by someone other than its author.

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T048 Run every scenario in [quickstart.md](./quickstart.md) end to end against a published tag and record the results
- [X] T049 Run the consuming-project integration check (quickstart Scenario 6) on a spare port alongside the existing `mlapi` container. Useful, **not authoritative** — Constitution Development Workflow §4. Note that its log-grep step is superseded by Scenario 5
- [X] T050 Record the cold-build wall-clock time and the warm (cached) time in `README.md` or a build note, and state which `CUDA_ARCH_BIN` list produced them. This is the project's live schedule risk against the 6-hour cap; if a cold build cannot fit, follow the escalation order in [research.md](./research.md) R8 — and its last two steps are the operator's decision, not an agent's. While measuring, also time **one cold build at a single architecture** (`CUDA_ARCH_BIN=6.1`). Two numbers solve for both halves of R8's `A + N×K`: the single-arch build is `A + K` and the four-arch build is `A + 4K`, which is the only thing that says whether per-architecture images (R8 option 3) would actually buy anything or merely quadruple the CI surface. Answer it before the CI work rather than after: option 3 would rewrite the workflow matrix, the container contract and two README sections, so taking it late is rework to absorb — the same reason this measurement is front-loaded in the first place
- [X] T050a Verify build-level reproducibility (SC-007): build the same source revision twice with the same build arguments and confirm both images report the same OpenCV version and the same CUDA build configuration from `scripts/assert-cuda-build.py`, and load models of the same versions. Record the one honest exception — the exported `.onnx` is not bit-reproducible, so compare the model *inputs* by checksum and the exported artifacts by what they load rather than by digest. T025 compares two containers from one tag, which is SC-014 and a different claim: this one is what lets a bug report distinguish "the image changed" from "my setup changed"
- [X] T051 Review every principle in `.specify/memory/constitution.md` v1.1.0 against the delivered artifacts and record the result, including two checks that are otherwise nobody's job: that the shipped image contains **no ZoneMinder awareness** — no ZM API client, credentials, or event/monitor concepts in the installed packages or our own code (FR-006, Principle III) — and that no build input resolves to a floating reference (SC-006, Principle II). Compliance is checked before each release, and a violation is either corrected or justified in writing
- [ ] T052 Cut `v0.1.0`: push the tag, let the pipeline verify it, then **pull it on the GPU host and re-run quickstart Scenarios 4 and 5 before announcing it**. CI proves the build; only hardware proves the runtime, and a release is not confirmed until that has happened (Development Workflow §3)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: no dependencies
- **Foundational (Phase 2)**: needs Setup; **blocks everything**
- **US1 (Phase 3)**: needs Foundational. This is the MVP and, unusually for this template, it also
  gates the CI stories — there must be something to build
- **US2 (Phase 4)** and **US3 (Phase 5)**: both need US1's Dockerfile. Independent of each other
- **US4 (Phase 6)**: needs US1 to document, and T046 needs a published image from US2 or US3.
  Note that T034 also writes to `README.md` despite living in Phase 4 — the release-note convention
  belongs with the release machinery it governs, not with the user-facing documentation pass
- **Polish (Phase 7)**: needs everything

### Why the CI stories are not fully independent

The template's ideal is stories that can be built in any order. Here US2 and US3 genuinely cannot
precede US1 — a workflow with no Dockerfile has nothing to publish. This is stated rather than
disguised. What they *are* is independently **testable**: once the Dockerfile exists, either can be
built and verified without the other.

### Within User Story 1

Most of Phase 3 edits one file, `Dockerfile`, and is therefore sequential: T009 → T010 → T011 →
T013 → T014 → T015 → T016 → T017 → T018 → T019. T012 (`scripts/export-onnx.py`) and T020–T022
(`entrypoint.sh`) are separate files. Verification tasks T024–T027 (including T024a, T025a and
T026a) need the build complete; T027 additionally needs GPU hardware. T025a and T026a deliberately
do not — both run under `PYZM_SERVE_ALLOW_CPU=1`, because self-containment and endpoint shape are
claims about the image, not about the hardware under it.

### Parallel Opportunities

- **Phase 1**: T003 and T004 — two different vendored files
- **Phase 2**: T008 (`scripts/assert-cuda-build.py`) runs alongside T005/T006/T007
- **Phase 3**: T012 (`scripts/export-onnx.py`) alongside the early Dockerfile stages; `entrypoint.sh`
  (T020–T022) alongside T009–T019 since it touches no shared file
- **Phase 4**: T029 (`scripts/verify-published.sh`) before or alongside the workflow itself; T030a
  (the immutability guard) is a self-contained job the build job `needs:`, so it can be written
  independently of T031; T034
  (the README **Releasing** section) touches a different file from the workflow and is independent
  of every other Phase 4 task
- **Phase 6**: T041 can start any time; the rest of the README needs the image to describe
- **Across phases**: US2 and US3 are parallelisable once US1 lands

### Parallel Example: Phase 2 + early Phase 3

```bash
# Different files, no shared dependencies:
Task: "T008 Write scripts/assert-cuda-build.py"
Task: "T012 Write scripts/export-onnx.py"
Task: "T020 Write entrypoint.sh env-to-flag mapping"
```

---

## Implementation Strategy

### MVP First (User Story 1)

1. Phase 1 Setup → Phase 2 Foundational → Phase 3 US1
2. **Stop and validate**: T024 and T024a (both gates bite), T025 and T025a (self-contained — and
   actually serving offline), T026 (fails loudly), T026a (surface unchanged), T027 (works on the
   GPU)
3. At this point there is a working, hand-buildable image. That is already worth having

### Incremental Delivery

1. Setup + Foundational → every input pinned, the assertion written
2. US1 → a working image (**MVP**)
3. US2 → immutable verified releases anyone can pull
4. US3 → mistakes caught on push rather than at release
5. US4 → usable by someone other than the author
6. Polish → `v0.1.0`, confirmed on hardware

### Front-load the schedule risk

Run one full cold build (T050's measurement) **as soon as T015 lands**, before writing any workflow.
The OpenCV CUDA compile across four architectures against a 6-hour cap is the one unknown in this
plan that no amount of design resolves, and finding out it does not fit is much cheaper before the
CI work than after it.

---

## Notes

- Commit after each task or logical group; commit messages open with a one-sentence summary followed
  by the reasoning (Constitution, Development Workflow)
- Work lands directly on `main`. Do not create a feature branch — Development Workflow §1 forbids an
  agent introducing one on its own initiative
- Three verification tasks need real GPU hardware and cannot be completed by CI or an agent: T027,
  parts of T048, and T052. Everything else — including the two gate-bites-checks (T024, T024a), the
  offline serve (T025a) and the endpoint-surface check (T026a) — runs on any machine with Docker
- Tasks carrying a letter suffix (T024a, T025a, T026a, T030a, T050a) were added by
  `/speckit-analyze` after the first pass found the requirements they cover had no task at all. The
  suffix keeps every existing task ID stable rather than renumbering the list
- On genuine confusion or an unplanned significant decision — the build not fitting the runner cap
  being the likeliest — stop and ask the operator rather than guessing (Development Workflow §5)
