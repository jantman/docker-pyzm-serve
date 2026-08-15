# Quickstart & Validation: GPU-Accelerated pyzm.serve Inference Gateway Image

**Feature**: `001-pyzm-serve-gpu-image` | **Date**: 2026-08-15

Runnable scenarios that prove the feature works. Each maps to acceptance criteria in
[spec.md](./spec.md) and is written to be executed, not read. Scenarios 1–3 need no GPU; 4–6 do.

Replace `<owner>` with the GitHub owner and `<tag>` with the tag under test throughout.

---

## Prerequisites

| Scenario | Needs |
|---|---|
| 1–3 | Docker with BuildKit. No GPU. |
| 4–6 | An NVIDIA GPU of compute capability ≥ 6.1, driver ≥ 550, NVIDIA Container Toolkit. |
| 7 | A GitHub fork with Actions enabled. |

A sample image is needed for inference scenarios. Any photo containing a person or a car works:

```bash
curl -sL -o sample.jpg https://ultralytics.com/images/bus.jpg
```

---

## Scenario 1 — The build refuses to ship a CUDA-less OpenCV

*Validates FR-008, FR-009, SC-003. Constitution Principle I.*

```bash
docker build -t pyzm-serve:local .
```

**Expected**: the build completes, and its log shows the CUDA assertion stage passing.

Now prove the gate actually bites — this is the test that matters, because a gate nobody has seen
fail is not known to work:

```bash
# Temporarily set the OpenCV CUDA flag off and confirm the build FAILS.
docker build --build-arg OPENCV_CUDA=OFF -t pyzm-serve:should-fail . ; echo "exit=$?"
```

**Expected**: non-zero exit, failing at `scripts/assert-cuda-build.py` with a message stating that
the OpenCV in the image reports no CUDA support. Nothing is tagged.

> Run this deliberately at least once per significant Dockerfile change. The predecessor's year-long
> CPU bug existed because nobody ever checked that the check worked.

---

## Scenario 2 — The image is self-contained

*Validates FR-033, FR-034, SC-013. Constitution Principle VI.*

```bash
docker run --rm --network none pyzm-serve:local \
  python3 -c "import os; p='/var/lib/zmeventnotification/models'; \
print(sorted(f for _,_,fs in os.walk(p) for f in fs))"
```

**Expected**: `yolo11m.onnx`, `yolo11s.onnx`, `yolov4.weights`, `yolov4.cfg`, `coco.names` — with
no network and no volumes.

Confirm no build toolchain shipped (FR-018, SC-008):

```bash
docker run --rm pyzm-serve:local sh -c 'which gcc g++ cmake nvcc || echo "no toolchain: correct"'
docker image inspect pyzm-serve:local --format '{{.Size}}' | numfmt --to=iec
```

**Expected**: `no toolchain: correct`, and a size to record in the README.

Confirm reproducibility of the environment across containers (FR-034, SC-014):

```bash
for i in 1 2; do
  docker run --rm pyzm-serve:local sh -c \
    'find /var/lib/zmeventnotification/models -type f | sort | xargs sha256sum'
done | sort | uniq -c
```

**Expected**: every line counted exactly twice — identical model sets and checksums.

---

## Scenario 3 — GPU is required, and its absence is loud

*Validates FR-004, RC-2, exit code 78. Constitution Principle I.*

```bash
docker run --rm pyzm-serve:local ; echo "exit=$?"
```

**Expected**: `exit=78`, with a message stating that GPU inference was requested but no CUDA device
is visible, naming `--gpus all` as the fix. **The container must not start and serve CPU
inference.** A silent CPU fallback here is the single failure this project exists to prevent.

The escape hatch still works:

```bash
docker run --rm -e PYZM_SERVE_ALLOW_CPU=1 -e PYZM_SERVE_PROCESSOR=cpu \
  -p 5000:5000 pyzm-serve:local
```

**Expected**: starts, warns loudly that it is running on CPU, and serves.

Auth refuses an unsafe default (FR-007, RC-3):

```bash
docker run --rm -e PYZM_SERVE_AUTH=1 pyzm-serve:local ; echo "exit=$?"
```

**Expected**: `exit=78`, refusing to sign tokens with upstream's published `change-me` secret.

---

## Scenario 4 — Serve detections on the GPU

*Validates User Story 1, FR-001 to FR-003, SC-001. Needs a GPU.*

```bash
docker run -d --gpus all -p 5001:5000 --name pyzm-test pyzm-serve:local
```

Health and model listing:

```bash
curl -sf localhost:5001/health          # {"status":"ok","models_loaded":true}
curl -sf localhost:5001/models | python3 -m json.tool
```

**Expected**: `yolo11m` present, `"loaded": true`. `yolo11s` and `yolov4` absent — present on disk
but not requested (MA-2).

Inference:

```bash
curl -sf -F type=object -F image=@sample.jpg localhost:5001/infer | python3 -m json.tool
```

**Expected**: a `detections` array with `label`, `confidence`, `box`, `type` and
`model_name: yolo11m`; `"error": null`.

Latency (SC-001, ≤ 300 ms p95):

```bash
for i in $(seq 1 20); do
  curl -so /dev/null -w '%{time_total}\n' -F type=object -F image=@sample.jpg localhost:5001/infer
done | sort -n | tail -2
```

**Expected**: the 19th value (p95 of 20) at or under 0.300 s on Pascal-class hardware.

Select the lighter model without rebuilding (FR-003, MA-2):

```bash
docker run --rm --gpus all -p 5002:5000 -e PYZM_SERVE_MODELS=yolo11s pyzm-serve:local &
sleep 30 && curl -sf localhost:5002/models | grep -q yolo11s && echo "alternative model OK"
```

---

## Scenario 5 — Prove the GPU is actually being used

*Validates FR-010 to FR-012, SC-002. Constitution Principle I. **This is the important one.***

Three checks. None substitutes for the others.

**5a — compiled with CUDA** (works without a GPU; proves what was built):

```bash
docker run --rm pyzm-serve:local python3 -c \
  "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'CUDA' in l])"
```

**Expected**: a line reporting NVIDIA CUDA as `YES`.

**5b — a CUDA device is visible** (needs a GPU; proves what is available):

```bash
docker run --rm --gpus all pyzm-serve:local python3 -c \
  "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
```

**Expected**: a non-zero device count. Zero means the GPU is not reaching the container.

**5c — the GPU is doing the work** (proves what is *used* — the only check that does):

```bash
# Terminal 1
nvidia-smi dmon -s u
# Terminal 2 — sustained load
for i in $(seq 1 100); do
  curl -so /dev/null -F type=object -F image=@sample.jpg localhost:5001/infer
done
```

**Expected**: non-zero GPU utilisation in `dmon` throughout the 100 requests.

**Also check the logs — but do not trust them alone**:

```bash
docker logs pyzm-test 2>&1 | grep -iE "fell back|does not support CUDA"   # expect empty
```

> ⚠️ Empty output here is **not** proof of GPU execution. Upstream's `_setup_gpu()` checks only the
> OpenCV *version*, and `setPreferableBackend(DNN_BACKEND_CUDA)` succeeds on a CPU-only build, so a
> CUDA-less image produces exactly this silence while running every frame on the CPU
> ([research R6](./research.md)). This grep catches genuine runtime CUDA errors and nothing else.
> 5a + 5b + 5c together are the proof.

---

## Scenario 6 — Consuming-project integration check

*Validates the handoff in the consuming project's `contracts/container-images.md`. Not this
project's acceptance surface (Constitution, Development Workflow §4) — a useful real-world test.*

```bash
docker run -d --gpus all -p 5001:5000 --name pyzm-test ghcr.io/<owner>/docker-pyzm-serve:<tag>
curl -sf localhost:5001/health
curl -sf localhost:5001/models | grep -q yolo11m
curl -sf -F type=object -F image=@sample.png localhost:5001/infer
```

Runs on a spare port alongside the existing `mlapi` container, so it is verifiable without touching
a running detection stack. Note the contract's log-grep step is superseded by Scenario 5.

---

## Scenario 7 — Publish and verify a release

*Validates User Stories 2 and 3, FR-021 to FR-027, SC-004. Needs a fork with Actions enabled.*

```bash
git push origin main                       # build.yml
git tag v0.0.1-test && git push origin v0.0.1-test   # release.yml
```

**Expected**:

1. `build.yml` publishes `ghcr.io/<owner>/docker-pyzm-serve:main-build<run_id>-<sha>` — a tag no one
   could mistake for a release.
2. `release.yml` publishes `v0.0.1-test`, and a **separate job** pulls that tag back from GHCR and
   re-runs the CUDA assertion against it.
3. A GitHub release is created.
4. No Docker Hub push occurs, and no repository secret was needed.

Confirm provenance on the published image (FR-025):

```bash
docker buildx imagetools inspect ghcr.io/<owner>/docker-pyzm-serve:v0.0.1-test --format '{{json .Provenance}}'
docker image inspect ghcr.io/<owner>/docker-pyzm-serve:v0.0.1-test \
  --format '{{json .Config.Labels}}' | python3 -m json.tool
```

**Expected**: the four OCI labels with the correct source, revision and version, plus SBOM and
provenance attestations.

Confirm immutability (FR-024, PIm-1) by re-running the release workflow for the existing tag and
verifying the digest is unchanged:

```bash
docker buildx imagetools inspect ghcr.io/<owner>/docker-pyzm-serve:v0.0.1-test --format '{{.Manifest.Digest}}'
```

---

## Scenario 8 — A stranger can adopt it

*Validates User Story 4, FR-029 to FR-032, SC-005, SC-012.*

Hand the README to someone with a GPU host and no knowledge of this project. They should reach a
running, GPU-verified gateway in under 30 minutes using only copy-pasteable commands, without asking
a question.

Checks that do not need a volunteer:

```bash
grep -riE "jantman|192\.168\.|10\.0\.|bigserver|/home/" README.md docker-compose.yml Dockerfile entrypoint.sh
```

**Expected**: no matches — no author-specific values anywhere (FR-032, SC-012).

The README must also contain: a repostatus badge; the best-effort support statement; the supported
GPU generations; the models shipped; the full environment-variable table; the image download size;
the two defaults that differ from upstream; and the Scenario 5 verification commands verbatim.

---

## Release confirmation gate

Per Constitution Development Workflow §3, a release is **not confirmed** until Scenarios 4 and 5
have passed on real hardware against the published tag. CI proves the build; only a GPU proves the
runtime. Do not announce a release that has only been verified by CI.
