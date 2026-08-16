# docker-pyzm-serve

[![Project Status: WIP – Initial development is underway, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

A container image that runs [pyzmNg](https://github.com/ZoneMinder/pyzmNg)'s `pyzm.serve` ML
detection gateway **on an NVIDIA GPU**, with OpenCV compiled from source with CUDA and cuDNN,
and every model baked in.

## Support and expectations

This is a **public, personal, best-effort open source project**. It is maintained by one
person for their own use, and it is genuinely intended to be usable by anyone else running
ZoneMinder with an NVIDIA GPU.

Both halves of that are meant literally:

- Issues may not be addressed. Pull requests are welcome but are not guaranteed a review.
- There is no obligation to support hardware, operating systems, or use cases the maintainer
  cannot test. Declining is always acceptable; pretending is not.
- Nothing site-specific is baked in. If you find a hostname, an IP address, a path or a
  hardware assumption from someone else's network in this image, that is a bug — please
  report it.

Setting expectations is a feature, not an apology.

## Why this exists

The stack this replaces ran **every frame on the CPU for over a year** without anyone
noticing. A PyPI `opencv-python` wheel had silently shadowed a CUDA-enabled OpenCV build, and
pyzm went on cheerfully logging `processor: gpu` for inference that had fallen back. The only
symptom was that detection felt a bit slow.

That failure is worse for a stranger than it was for its author: you have no reason to suspect
it, and nothing in the logs will tell you. So this image is built so it **cannot** happen:

| Layer | Check | Where it runs |
|---|---|---|
| Build | `cv2.getBuildInformation()` must report CUDA, and `cv2.cuda` must exist | a `RUN` in the Dockerfile — fails the build |
| Publish | the same assertion, against the image **pulled back from the registry** | a separate CI job after the push |
| Runtime | a CUDA device is visible, and `nvidia-smi` shows work happening | **you**, on your hardware — see below |
| Start-up | the container refuses to start if you asked for GPU and there is none | the entrypoint, exit code `78` |

The first two need no GPU, which is what makes them runnable in CI. Only the third proves
execution, and only you can run it.

---

## Prerequisites

- An NVIDIA GPU of **compute capability 6.1 or newer** (GTX 10-series / Pascal onwards)
- An NVIDIA driver of the **550 branch or newer**
- The [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker

Check the first three:

```bash
nvidia-smi                                    # driver version, and your GPU
docker run --rm --gpus all ubuntu nvidia-smi  # the toolkit is wired into Docker
```

If the second command fails, fix that before going any further — the image will refuse to
start without it, on purpose.

## Quickstart

```bash
docker run -d --gpus all -p 5000:5000 --name pyzm-serve \
  ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

Wait for it to become healthy (first start loads the model; see
[GPU generations](#gpu-generations) if yours is newer than Ada, which adds a one-off delay):

```bash
docker inspect --format '{{.State.Health.Status}}' pyzm-serve
curl -sf localhost:5000/health          # {"status":"ok","models_loaded":true}
curl -sf localhost:5000/models | python3 -m json.tool
```

Run a detection:

```bash
curl -sL -o sample.jpg https://ultralytics.com/images/bus.jpg
curl -sf -F type=object -F image=@sample.jpg localhost:5000/infer | python3 -m json.tool
```

You should get a `detections` array with `label`, `confidence`, `box` and
`model_name: yolo11m`.

### With Docker Compose

[`docker-compose.yml`](./docker-compose.yml) in this repository is a complete worked example.
Copy it, set the image tag to the release you want, and:

```bash
docker compose up -d
docker compose ps          # STATUS should reach (healthy)
docker compose logs -f
```

The GPU reservation is the part that matters, and it is the Compose equivalent of
`--gpus all`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**No volumes are needed.** The image ships every model it serves and requires no network
access after it is pulled. If you find yourself adding a volume to make it work at all, that
is a bug.

---

## Verify the GPU is actually being used

**Do this once, on your own hardware, before you trust it.** Three checks. None of them
substitutes for the others.

### 1. Compiled with CUDA (no GPU required — proves what was *built*)

```bash
docker run --rm --entrypoint python3 ghcr.io/jantman/docker-pyzm-serve:v0.1.0 \
  -c "import cv2; print([l for l in cv2.getBuildInformation().splitlines() if 'CUDA' in l])"
```

Expect lines reporting **NVIDIA CUDA as `YES`** and **cuDNN as `YES`**. Both matter: CUDA
alone gives you the `cv2.cuda` namespace, but the DNN module — which is what actually runs
inference here — only gets its CUDA backend when cuDNN is present.

> `--entrypoint python3` is needed on any command like this. Without it, `python3 -c …` is
> appended to the *server's* command line (that is the documented behaviour — see
> [Configuration](#configuration)), and the container tries to start the gateway instead.

### 2. A CUDA device is visible (proves what is *available*)

```bash
docker run --rm --gpus all --entrypoint python3 ghcr.io/jantman/docker-pyzm-serve:v0.1.0 \
  -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
```

Expect a **non-zero** device count. Zero means the GPU is not reaching the container — check
the NVIDIA Container Toolkit and that you passed `--gpus all`.

### 3. The GPU is doing the work (proves what is *used* — the only check that does)

In one terminal:

```bash
nvidia-smi dmon -s u
```

In another, with the container running and `sample.jpg` to hand:

```bash
for i in $(seq 1 100); do
  curl -so /dev/null -F type=object -F image=@sample.jpg localhost:5000/infer
done
```

Expect **non-zero GPU utilisation** in `dmon` throughout.

### Also check the logs — but do not trust them alone

```bash
docker logs pyzm-serve 2>&1 | grep -iE "fell back|does not support CUDA"   # expect empty
```

> ⚠️ **Empty output here is NOT proof of GPU execution.**
>
> Upstream's `_setup_gpu()` checks only the OpenCV *version*, and
> `setPreferableBackend(DNN_BACKEND_CUDA)` succeeds even on a CPU-only build. A CUDA-less
> image produces exactly this silence while running every frame on the CPU. This grep catches
> genuine *runtime* CUDA errors and nothing else.
>
> Checks 1 + 2 + 3 together are the proof. This warning exists because a check that could
> pass while the GPU sits idle is not a check — and believing otherwise is what cost the
> predecessor a year.

---

## Configuration

Everything is an environment variable. There is no configuration file, by design.

| Variable | Default | Effect |
|---|---|---|
| `PYZM_SERVE_MODELS` | `yolo11m` | Models to load. Space-separated. `all` discovers everything in the base path (lazily loaded). |
| `PYZM_SERVE_PROCESSOR` | `gpu` | `gpu`, `cpu` or `tpu`. |
| `PYZM_SERVE_PORT` | `5000` | Listen port. The healthcheck follows this automatically. |
| `PYZM_SERVE_HOST` | `0.0.0.0` | Bind address. |
| `PYZM_SERVE_BASE_PATH` | `/var/lib/zmeventnotification/models` | Model tree root. |
| `PYZM_SERVE_WORKERS` | `1` | uvicorn workers. On a GPU, more workers multiply VRAM with no throughput gain. |
| `PYZM_SERVE_DEBUG` | unset | Any non-empty value enables debug logging. |
| `PYZM_SERVE_AUTH` | unset (off) | Any non-empty value enables JWT authentication. |
| `PYZM_SERVE_AUTH_USER` | `admin` | Auth username. |
| `PYZM_SERVE_AUTH_PASSWORD` | unset | Auth password. **Required when auth is on.** |
| `PYZM_SERVE_TOKEN_SECRET` | unset | JWT signing secret. **Required when auth is on.** |
| `PYZM_SERVE_ALLOW_CPU` | unset | Downgrades the "no GPU visible" refusal to a warning. Start-up only — see below. |
| `PYZM_SERVE_NO_CPU_FALLBACK` | unset | Any non-empty value makes a GPU failure fail the request instead of degrading to CPU. |
| `PYZM_SERVE_GPU_RETRY_SECONDS` | unset (upstream: `60`) | Seconds on CPU after a GPU failure before the GPU is retried, doubling to a 15-minute cap. `0` makes a fallback permanent. |
| `PYZM_SERVE_WARMUP` | `1` | `0` skips the start-up inference that proves the GPU actually runs a frame. |
| `PYZM_SERVE_WARMUP_TIMEOUT` | `300` | Seconds the warm-up waits for models to load before giving up. |

Anything you pass after the image name on `docker run` is appended verbatim to the server
command line, so this table is a convenience, never a boundary:

```bash
docker run --gpus all -p 5000:5000 ghcr.io/jantman/docker-pyzm-serve:v0.1.0 --debug
```

### Two defaults deliberately differ from upstream

If you are following upstream pyzm's documentation, these will surprise you. They are
different on purpose:

| Setting | Upstream default | This image | Why |
|---|---|---|---|
| `--processor` | `cpu` | **`gpu`** | This is a GPU image. A user who silently gets CPU has the predecessor's bug back as a feature. |
| `--models` | `yolo11s` | **`yolo11m`** | `yolo11m` is the primary model here; `yolo11s` is still available. |

### Authentication is OFF by default — what that means

With authentication off, **this gateway answers any request that can reach its port.** There
is no credential check of any kind.

That is a reasonable default for the normal deployment: a detection gateway on a trusted
network segment alongside the ZoneMinder host that calls it. It is a bad default for anything
else — a shared LAN, a VPS, a port forwarded through a router. A default is only safe if the
person relying on it knows what it assumes.

To turn authentication on, set **all three**:

```bash
docker run -d --gpus all -p 5000:5000 \
  -e PYZM_SERVE_AUTH=1 \
  -e PYZM_SERVE_AUTH_USER=admin \
  -e PYZM_SERVE_AUTH_PASSWORD='choose-a-real-password' \
  -e PYZM_SERVE_TOKEN_SECRET="$(openssl rand -hex 32)" \
  ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

The container **refuses to start** (exit `78`) if auth is enabled and either the password or
the token secret is missing. Upstream's `--token-secret` defaults to the literal string
`change-me`, which is published in its source; signing tokens with it would let anyone who has
read that source mint a valid token. An authenticated endpoint whose credentials come from
someone else's source tree is worse than an open one, because you believe it is protected.

### Exit codes

| Exit | Meaning |
|---|---|
| `0` | Normal shutdown |
| `1` | The server exited with an error |
| `78` | **Configuration error** — GPU requested with no CUDA device visible, or auth enabled without a password or token secret. The message names the fix. |

`78` is `EX_CONFIG` from `sysexits.h`, so a misconfiguration is distinguishable from a crash.

A failed start-up warm-up (see below) is the one stop that does **not** get its own code: it
signals the already-running server to shut down, so the container exits `0` as if you had
stopped it. The log says why, in a block beginning `WARM-UP FAILED`.

### When the GPU degrades

A CUDA error during inference does not always mean the GPU is broken. It can be a one-off:
on the deployment that produced [issue #1](https://github.com/jantman/docker-pyzm-serve/issues/1)
a single `CUDA-capable device(s) is/are busy or unavailable` on the first request moved
every subsequent frame onto the CPU — about 8x slower — for the life of the container,
while `/health` kept answering `{"status":"ok"}`. Nothing surfaced it.

Three things now stand between you and that:

**The container proves the GPU works before you depend on it.** At start-up, after the
models load, one synthetic frame is posted to `/infer` and the result is checked against
`/models`. If that frame does not come back from the processor you asked for, the container
logs why and stops, so your restart policy recreates it rather than serving degraded. This
is what makes a green healthcheck mean something on a container that has not yet had
traffic: until a frame has actually run, `/models` can only report what was *configured*.
Set `PYZM_SERVE_WARMUP=0` to skip it.

If `PYZM_SERVE_ALLOW_CPU` is set the warm-up still runs and still reports, but it will not
stop the container — you have already said you might not get a GPU, and a restart loop is no
way to be told so.

**A degraded container reports itself unhealthy.** `/models` now carries both the processor
each model is running on and the one it was asked for, and the healthcheck fails when they
differ:

```console
$ docker inspect --format '{{.State.Health.Status}}' pyzm-serve
unhealthy
$ curl -s localhost:5000/models | jq '.models[] | {name, processor, requested_processor}'
{ "name": "yolo11m", "processor": "cpu", "requested_processor": "gpu" }
```

Note that Docker will not act on that by itself — `restart: unless-stopped` does not react
to health status. It makes the condition visible and alertable; restarting on it needs
something like a watchdog container, or your monitoring.

**A transient fault heals itself.** After a fallback the GPU is retried 60 seconds later,
doubling after each further failure up to 15 minutes, so a glitch costs you a minute of
slow frames rather than an outage. A genuinely dead GPU is not re-probed on every request.
Tune with `PYZM_SERVE_GPU_RETRY_SECONDS`, or set it to `0` for the old permanent behaviour.

If you would rather fail than be slow, `PYZM_SERVE_NO_CPU_FALLBACK=1` makes `/infer` return
an error and keeps the model on the GPU. That suits a caller that can retry; it means a
missed detection rather than a late one, so it is not the default.

`PYZM_SERVE_ALLOW_CPU` has no bearing on any of this. It governs start-up only — whether a
container with no visible GPU refuses to start — and is deliberately not wired to the
runtime policy, so that "must start with a GPU" and "may degrade while running" stay
separate choices.

### Running without a GPU

Supported for trying the image out, not as a deployment:

```bash
docker run --rm -p 5000:5000 \
  -e PYZM_SERVE_PROCESSOR=cpu -e PYZM_SERVE_ALLOW_CPU=1 \
  ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

It will warn loudly and run roughly an order of magnitude slower per frame.

---

## What's in the image

### Models

All three ship inside the image. Nothing is downloaded at start-up.

| Request this name | Role | Format | Loaded by default |
|---|---|---|---|
| `yolo11m` | primary | ONNX (YOLO11 medium) | **yes** |
| `yolo11s` | lighter alternative | ONNX (YOLO11 small) | no — present on disk |
| `yolov4` | legacy fallback | Darknet | no — present on disk |

Select a different one without rebuilding:

```bash
docker run --gpus all -p 5000:5000 -e PYZM_SERVE_MODELS=yolo11s \
  ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

`GET /models` on a running container is the authoritative answer for what that image actually
serves. Names are file stems, resolved by upstream's discovery rules.

The YOLO11 models are exported to ONNX at build time from pinned Ultralytics `.pt` weights, at
`imgsz=640` to match what upstream's ONNX backend assumes. Their class labels come from
metadata embedded in the model. YOLOv4 uses the vendored `coco.names` alongside it.

### Supplying your own model

Supported — adding to the shipped set is a feature. Mount a **subdirectory** of the base path
so the shipped models stay in place, and name your model by its file stem:

```bash
docker run -d --gpus all -p 5000:5000 \
  -v /path/to/my/models:/var/lib/zmeventnotification/models/custom:ro \
  -e PYZM_SERVE_MODELS="yolo11m my-model" \
  ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

⚠️ Mounting over `/var/lib/zmeventnotification/models` itself **replaces** the shipped models
rather than supplementing them. That is a legitimate choice, but make it deliberately.

ONNX models should be exported at `imgsz=640` and without baked-in NMS, matching how the
shipped ones are built — otherwise detections are silently mis-scaled.

### GPU generations

| Compute capability | Generation | Example hardware | Status |
|---|---|---|---|
| 6.1 | Pascal | GTX 10-series, Quadro P1000 | ✅ Compiled in |
| 7.5 | Turing | GTX 16-series, RTX 20-series | ✅ Compiled in |
| 8.6 | Ampere (consumer) | RTX 30-series | ✅ Compiled in |
| 8.9 | Ada | RTX 40-series | ✅ Compiled in |
| newer than 8.9 | Blackwell and later | RTX 50-series | ⚠️ **Works via PTX JIT** — see below |
| older than 6.1 | Maxwell and earlier | GTX 9-series | ❌ **Not supported** — no fallback exists |

**If your GPU is newer than Ada**, the image ships forward-compatible intermediate code (PTX)
that your driver JIT-compiles on **first model load**. This works, and it can take tens of
seconds the first time. That is why the healthcheck has a 180-second start period — do not
mistake it for a hang.

**If your GPU is older than Pascal**, this image will not work for you and there is nothing to
configure. Inference would fail with "no kernel image is available for execution on the
device". Note that Pascal itself has a finite life: NVIDIA's 580 driver branch is the last to
support it, and its eventual removal here will be a breaking change, signalled in the version.

### Image size

<!-- MEASURED 2026-08-15 against the local build: 4363005636 bytes on disk. Re-measure when
     the base image, the model set or the OpenCV BUILD_LIST changes. -->
**4.1 GiB on disk** once pulled (about 4.36 GB / 2–3 GB of download, since layers transfer
compressed). That is large, and it is the direct price of two deliberate choices: a full
CUDA + cuDNN runtime, and baking in every model so the image needs no network after it is
pulled — you pay it once, at pull time, on a machine, rather than during a recovery, at the
worst possible moment, as a person.

Roughly: ~3.4 GB of CUDA 12.4 + cuDNN 9.1 runtime libraries from NVIDIA's base image,
~310 MB of models (`yolov4.weights` alone is 250 MB), ~180 MB of OpenCV and its Python
bindings, and the rest Python. **No compiler toolchain is included** — no `gcc`, no `cmake`,
no `nvcc`. You can check:

```bash
docker run --rm --entrypoint sh ghcr.io/jantman/docker-pyzm-serve:v0.1.0 \
  -c 'which gcc g++ cmake nvcc || echo "no toolchain: correct"'
```

---

## HTTP API

The HTTP surface is **upstream's, unaltered** — this image adds no endpoints and changes no
semantics. A client written against `pyzm.serve` works here unmodified.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "models_loaded": true}` |
| `GET` | `/models` | Available models, their load status, and the processor each is running on vs the one requested |
| `POST` | `/infer` | Run detection on an uploaded frame |
| `POST` | `/login` | Obtain a JWT (only meaningful with auth enabled) |

See the [pyzm.serve API reference](https://pyzmng.readthedocs.io/en/latest/guide/serve.html)
for request and response schemas.

### What this image does not do

Stated so nobody plans around it: no ZoneMinder API client, credentials, or event/monitor
concepts; no frame selection, zone filtering, nuisance filtering, confidence policy,
past-detection matching or notification; no configuration file; no added or altered HTTP
endpoints; no run-time model downloads; no platform other than `linux/amd64`.

All of those belong to the client, by upstream's design. The server is a dumb inference
engine, deliberately.

---

## Versions and pinning

Images are published to **GitHub Container Registry only**. There is no Docker Hub image.

| Tag pattern | Deployable? | What it is |
|---|---|---|
| `v<semver>` | ✅ **yes — the only recommended target** | An immutable release. Never moved, never deleted. |
| `<digest>` | ✅ yes, strongest | Pin `…@sha256:…` if you want certainty beyond a tag. |
| `main-build<run_id>-<sha>` | ⚠️ not recommended | A build of a `main` commit. Not re-verified, no release notes. |
| `latest` | ❌ **never** | Moves on every release. Convenience only. |
| `buildcache` | ❌ not an image | BuildKit layer cache. Not runnable. |

**Do not deploy `latest`.** A published release tag always refers to the same image, for as
long as anything might pull it — so pin one:

```yaml
image: ghcr.io/jantman/docker-pyzm-serve:v0.1.0
```

Available releases: <https://github.com/jantman/docker-pyzm-serve/releases>

## Releasing

*(For maintainers, and for anyone forking this.)*

A release is cut by **pushing a git tag**. The image tag equals the git tag. There is no
manual publish step and **no repository secret to configure** — `GITHUB_TOKEN` authenticates
to GHCR, so a fork can build and publish its own images with no setup at all.

```bash
git tag v0.2.0
git push origin v0.2.0
```

`release.yml` then, in order: refuses to proceed if that tag is already published, builds and
pushes, **pulls the image back from GHCR and re-runs the CUDA assertion against it**, and only
then creates the GitHub release.

### Release note conventions

Release notes MUST explicitly call out any of these four breaking changes, together with the
version bump that signals it:

1. **A model removed** from the shipped set
2. **A default changed** (anything in the configuration table above)
3. **A GPU generation dropped** from `CUDA_ARCH_BIN`
4. **A new required setting** — anything that makes a previously working configuration fail

A user must be able to judge **from the tag alone** whether an upgrade is safe. That is what
the version number is for; if the notes are the only place a breaking change appears, the
version is wrong.

### Two rules that are not negotiable

- **A published tag is never moved or deleted.** People pin tags in compose files and Puppet
  manifests and have no way to know one moved under them; deleting one breaks a rollback path
  for people you will never hear from. The release workflow enforces this with a guard job
  that fails before building if the tag already exists.
- **A release is not confirmed until it has been pulled and verified on real GPU hardware.**
  CI proves the build; only a GPU proves the runtime. Do not announce a release that has only
  been verified by CI.

## Build notes

Building from source is not required to use this image, but if you want to:

```bash
docker build -t pyzm-serve:local .
```

Everything is pinned — base images by digest, git checkouts by full commit SHA, model weights
by SHA-256 checksum verified during the build. Overridable pins are `ARG`s at the top of the
[`Dockerfile`](./Dockerfile).

The dominant cost is compiling OpenCV with CUDA. Measured 2026-08-15 on a 24-core
workstation, with the ccache and package caches cleared before each cold run:

<!-- MEASURED: re-measure when CUDA_ARCH_BIN, BUILD_LIST or the OpenCV version changes.
     Both cold runs verified genuinely cold: ccache reported 0 hits / 648 misses. -->

| Build | `CUDA_ARCH_BIN` | Total | OpenCV compile stage |
|---|---|---|---|
| Cold, all four architectures | `6.1;7.5;8.6;8.9` | **5m03s** | 174.9s |
| Cold, single architecture | `6.1` | **4m52s** | 114.2s |
| Warm, no changes | — | **4s** | cached |

Two useful conclusions fall out of those two compile numbers. Solving
`A + 4K = 174.9` and `A + K = 114.2` gives an architecture-independent compile of
**A ≈ 94s** and a per-architecture cost of only **K ≈ 20s**.

- **Build time is a non-issue.** A cold build is ~5 minutes here, against a GitHub-hosted
  runner's 6-hour job cap. Even allowing for a standard runner being several times slower,
  there is a very large margin. The trimmed `BUILD_LIST` is what bought that — it compiles
  ten OpenCV modules rather than sixty.
- **Publishing one image per GPU architecture would not be worth it.** It was considered as
  a way to shorten the critical path. Since `A` dominates `K`, it would cut the compile from
  175s to 114s — while quadrupling the CI and verification surface and forcing every user to
  map their GPU to a compute capability and pick the matching tag by hand. Not taken.

Narrowing `CUDA_ARCH_BIN` is still the main lever if you only have one kind of GPU and want
a smaller image:

```bash
docker build --build-arg CUDA_ARCH_BIN=8.6 --build-arg CUDA_ARCH_PTX=8.6 -t pyzm-serve:local .
```

## Related projects

- [ZoneMinder/pyzmNg](https://github.com/ZoneMinder/pyzmNg) — the upstream this image packages
- [jantman/docker-zm-mlapi](https://github.com/jantman/docker-zm-mlapi) — the predecessor this
  replaces, whose Dockerfile still carries the TODO that started this project

## License

See [LICENSE](./LICENSE).
