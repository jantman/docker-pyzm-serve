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

### Why several commands below pass `--entrypoint`

Any scenario that runs *something other than the gateway* inside the image must override the
entrypoint — `docker run … IMAGE python3 -c …` does **not** work. Two reasons, both by design:

1. Trailing arguments are appended to the server command line, not executed (contract §2,
   RC-4), so `python3 -c …` would reach `python -m pyzm.serve` as flags.
2. The entrypoint's GPU preflight runs first and exits `78` on a machine with no GPU.

`--entrypoint python3` / `--entrypoint sh` bypasses both. This was found by executing these
scenarios rather than reading them.

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

The **checksum** gate deserves the same treatment (FR-014):

```bash
# Corrupt one recorded checksum, confirm the build refuses, then restore it.
cp models/checksums.sha256 /tmp/checksums.bak
sed -i '1s/^./0/' models/checksums.sha256
docker build -t pyzm-serve:should-fail . ; echo "exit=$?"
cp /tmp/checksums.bak models/checksums.sha256
```

**Expected**: non-zero exit, failing at the model verification step with a message naming the
artifact whose checksum did not match. A model that changes silently produces detection differences
indistinguishable from a code regression, and this gate is the only thing between a user and that.

---

## Scenario 2 — The image is self-contained

*Validates FR-013 (serves offline), FR-018, FR-033, FR-034, SC-013, SC-014. Constitution
Principle VI.*

```bash
docker run --rm --network none --entrypoint python3 pyzm-serve:local \
  -c "import os; p='/var/lib/zmeventnotification/models'; \
print(sorted(f for _,_,fs in os.walk(p) for f in fs))"
```

**Expected**: `yolo11m.onnx`, `yolo11s.onnx`, `yolov4.weights`, `yolov4.cfg`, `coco.names` — with
no network and no volumes.

Confirm no build toolchain shipped (FR-018, SC-008):

```bash
docker run --rm --entrypoint sh pyzm-serve:local -c 'which gcc g++ cmake nvcc || echo "no toolchain: correct"'
docker image inspect pyzm-serve:local --format '{{.Size}}' | numfmt --to=iec
```

**Expected**: `no toolchain: correct`, and a size to record in the README.

Confirm reproducibility of the environment across containers (FR-034, SC-014):

```bash
for i in 1 2; do
  docker run --rm --entrypoint sh pyzm-serve:local -c \
    'find /var/lib/zmeventnotification/models -type f | sort | xargs sha256sum'
done | sort | uniq -c
```

**Expected**: every line counted exactly twice — identical model sets and checksums.

### 2c — it actually *serves* offline

The checks above prove the files are present. This one proves the image does its job with nothing
outside it, which is what Principle VI actually claims. `--network none` makes published ports
impossible, so drive it from inside the container (there is no `curl` in the runtime image, so use
Python):

```bash
docker run -d --name pyzm-offline --network none \
  -e PYZM_SERVE_PROCESSOR=cpu -e PYZM_SERVE_ALLOW_CPU=1 pyzm-serve:local
docker cp sample.jpg pyzm-offline:/tmp/sample.jpg

docker exec pyzm-offline python3 - <<'PY'
import json, time, urllib.request

for _ in range(60):                                    # become healthy
    try:
        h = urllib.request.urlopen("http://127.0.0.1:5000/health").read()
        print("health:", h); break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit("never became healthy with no network")

b = b"--X\r\n"                                          # answer an inference request
b += b'Content-Disposition: form-data; name="type"\r\n\r\nobject\r\n--X\r\n'
b += b'Content-Disposition: form-data; name="image"; filename="s.jpg"\r\n\r\n'
b += open("/tmp/sample.jpg", "rb").read() + b"\r\n--X--\r\n"
r = urllib.request.Request("http://127.0.0.1:5000/infer", data=b,
                           headers={"Content-Type": "multipart/form-data; boundary=X"})
print("infer:", json.load(urllib.request.urlopen(r)))
PY

docker rm -f pyzm-offline
```

**Expected**: `{"status": "ok", "models_loaded": true}` followed by a `detections` array — with no
network and no volumes. CPU is correct here: the claim under test is self-containment, not GPU
execution.

> This is the test Principle VI names for itself — *"The image MUST start, become healthy, and
> answer an inference request with no network access and no volumes mounted. This is the test of
> this principle, and it is cheap enough to run every time."* Listing model files offline is not
> that test. A tag that needs the network to start is a recipe rather than a version, and it fails
> precisely during a recovery, when it is least affordable.

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

The HTTP surface must be exactly upstream's — nothing added, removed, or renamed (FR-002, SC-010,
Principle III). This check needs no GPU; run it against any started container:

```bash
curl -sf localhost:5001/openapi.json | python3 -c \
  "import json,sys; print(sorted(json.load(sys.stdin)['paths']))"
```

**Expected**: exactly the upstream routes — health, model listing, inference and login — and no
others. SC-010 promises that a client written against upstream works here unmodified; this is the
only thing that checks it, and a negative requirement nobody checks is a hope.

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
docker run --rm --entrypoint python3 pyzm-serve:local \
  -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'CUDA' in l])"
```

**Expected**: a line reporting NVIDIA CUDA as `YES`.

**5b — a CUDA device is visible** (needs a GPU; proves what is available):

```bash
docker run --rm --gpus all --entrypoint python3 pyzm-serve:local \
  -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
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

Confirm immutability (FR-024, PIm-1). Record the digest, re-run the release workflow for the tag
that already exists, and check both that the run **failed at the guard job** and that the digest did
not move:

```bash
docker buildx imagetools inspect ghcr.io/<owner>/docker-pyzm-serve:v0.0.1-test --format '{{.Manifest.Digest}}'
# re-run release.yml for v0.0.1-test from the Actions UI, then:
docker buildx imagetools inspect ghcr.io/<owner>/docker-pyzm-serve:v0.0.1-test --format '{{.Manifest.Digest}}'
```

**Expected**: the workflow stops at the tag-immutability guard before building anything, and the two
digests are identical. Note that the guard is what makes this pass — this build is not
bit-reproducible, so without it a re-run would push a *different* image to the same tag and the
digests would differ. GHCR accepts that silently.

---

## Scenario 8 — A stranger can adopt it

*Validates User Story 4, FR-029 to FR-032, SC-005, SC-012.*

Hand the README to someone with a GPU host and no knowledge of this project. They should reach a
running, GPU-verified gateway in under 30 minutes using only copy-pasteable commands, without asking
a question.

Checks that do not need a volunteer:

```bash
# The whole tracked tree, not a hand-picked handful of files.
grep -rIiE "192\.168\.|10\.0\.|bigserver|/home/" \
  --exclude-dir=.git --exclude-dir=specs .

# The image's own ENV. No exemptions here: nothing in the runtime environment should
# ever name a person, a host or a path from the author's network.
docker image inspect "$IMAGE" --format '{{json .Config.Env}}' \
  | grep -iE "jantman|192\.168\.|10\.0\.|bigserver|/home/" && echo "LEAK IN ENV" || echo "env clean"

# The image's labels, EXEMPTING the two OCI labels that are required to name the
# source repository. See the note below for why this exemption is not a loophole.
docker image inspect "$IMAGE" --format '{{json .Config.Labels}}' | python3 -c '
import json, re, sys
labels = json.load(sys.stdin) or {}
exempt = {"org.opencontainers.image.source", "org.opencontainers.image.url"}
pat = re.compile(r"jantman|192\.168\.|10\.0\.|bigserver|/home/", re.I)
bad = {k: v for k, v in labels.items() if k not in exempt and pat.search(str(v))}
print("LEAK IN LABELS:", bad) if bad else print("labels clean")'
```

**Expected**: no file matches, `env clean`, and `labels clean` (FR-032, SC-012).

**Two exemptions, and why neither is a loophole.**

The owner name is not an author-specific *value* in FR-032's sense — that clause is about
hostnames, IP addresses, local paths, credentials and hardware assumptions from the author's
network. It is the project's public identity, and two places are *required* to carry it:

- `org.opencontainers.image.source` and `.url` **must** point at the source repository. FR-025
  mandates those labels; a published image that omitted them would violate Principle IV. So the
  label check exempts exactly those two keys — and only those two, by name, so a hostname
  hiding in any other label is still caught.
- `README.md` and `docker-compose.yml` must give a **copy-pasteable** `ghcr.io/<owner>/...`
  reference, because FR-029 and SC-005 promise a stranger can run this from the README alone.
  A `<owner>` placeholder would fail that promise. The tracked-tree grep therefore drops the
  owner name from its pattern and keeps every genuinely site-specific pattern.

This was found by running the check against a real published image rather than a local build:
the local build has no OCI labels at all, so the naive one-liner passed locally and failed on
the artifact users actually pull. A check that cannot pass on a compliant artifact gets waved
through, and then it is protecting nothing — the same reasoning that produced constitution
v1.2.0.

The README must also contain: a repostatus badge; the best-effort support statement; the supported
GPU generations; the models shipped; the full environment-variable table; the image download size;
the two defaults that differ from upstream; and the Scenario 5 verification commands verbatim.

---

## Release confirmation gate

Per Constitution Development Workflow §3, a release is **not confirmed** until Scenarios 4 and 5
have passed on real hardware against the published tag. CI proves the build; only a GPU proves the
runtime. Do not announce a release that has only been verified by CI.
