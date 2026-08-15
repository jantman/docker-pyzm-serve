# Contract: CI/CD pipeline

**Feature**: `001-pyzm-serve-gpu-image` | **Date**: 2026-08-15

What the pipeline promises, to the maintainer and to anyone pulling a tag. Derived from
`docker-zm-mlapi`'s shape, with Docker Hub removed and post-publish verification added.

---

## 1. Workflows

| | `build.yml` | `release.yml` |
|---|---|---|
| Trigger | push to `main`; `workflow_dispatch` | push of a tag |
| Publishes | `main-build<run_id>-<sha>` | `v<semver>` **and** `latest` |
| Mutable output? | no | `latest` only |
| Tag-already-exists guard | n/a — the run ID makes collision impossible | **yes — first job, blocks the build** |
| Build-time CUDA assertion | yes (in the Dockerfile) | yes |
| Pull-back verification | no (FR-010 scopes it to releases) | **yes — separate job** |
| GitHub release | no | yes |

Two files rather than one keyed on ref type: "publish something to try" and "make a permanent
promise" deserve visibly different code paths.

---

## 2. Permissions and credentials

```yaml
permissions:
  contents: write    # release.yml only — creating the GitHub release
  packages: write    # pushing to GHCR
```

**No repository secret is required.** `GITHUB_TOKEN` authenticates to GHCR. Removing Docker Hub
deleted the `DOCKERHUB_TOKEN` setup step the sibling repository's `release.yml` documents in its
header comment — a genuine simplification worth stating in the README, since it means a fork needs
no manual credential setup to build.

`build.yml` needs `packages: write` only.

---

## 3. Publishing

- Registry: `ghcr.io` exclusively (FR-021). No workflow may push anywhere else.
- Action: `docker/build-push-action@v6` behind `docker/setup-buildx-action`.
- `platforms: linux/amd64`. QEMU multi-arch is forbidden — an emulated CUDA compile would not
  finish inside any timeout.
- `sbom: true` and `provenance: mode=max` (FR-025).
- OCI labels on every published image (FR-025):

| Label | Value |
|---|---|
| `org.opencontainers.image.url` | `https://github.com/${{ github.repository }}` |
| `org.opencontainers.image.source` | `https://github.com/${{ github.repository }}` |
| `org.opencontainers.image.version` | `${{ github.ref_name }}` |
| `org.opencontainers.image.revision` | `${{ github.sha }}` |

---

## 4. Verification

Three gates. The first two run in CI on GPU-less runners; the third cannot and does not.

| Gate | Where | Fails what |
|---|---|---|
| Compiled-with-CUDA assertion | Dockerfile `RUN` — so a local `docker build` gets it too | the build; nothing is published |
| Pull-back re-assertion | `release.yml`, a **separate job** with `needs:` the build | the pipeline; the release is not presented as usable |
| Runtime GPU proof | the operator, on real hardware | the release's confirmation (Workflow §3) |

**The pull-back job must pull the published tag from GHCR** and run the assertion against that
image. Re-running it against build-local state would defeat FR-010, whose whole purpose is catching
a bad cache hit, a wrong tag, or a registry mixup — failures invisible from inside the build.

Both CI gates execute `scripts/assert-cuda-build.py`, the same file, so they cannot drift apart.

**Not a gate**: grepping container logs for `fell back` or `does not support CUDA`. That check is
retained in the operator documentation because it catches genuine runtime CUDA errors, but it
passes when OpenCV was built without CUDA at all (research R6) and must never be presented as
proof of GPU execution.

---

## 5. Caching and time budget

- `cache-from` / `cache-to`: `type=registry,mode=max`, ref `ghcr.io/<owner>/docker-pyzm-serve:buildcache`.
- Preferred over `type=gha`, whose 10 GB per-repository budget an OpenCV CUDA build exhausts.
- `buildcache` is mutable and is not a runnable image. It is neither a release nor deployable, so it
  does not engage FR-024 — but it is named so nobody mistakes it for one.
- Every job sets an explicit `timeout-minutes` below the 6-hour hard cap, so a runaway build fails
  as a timeout with a clear cause rather than being silently killed at the ceiling.
- Standard GitHub-hosted runners only. Larger runners require a paid plan the maintainer does not
  have; self-hosted runners are out of scope.

If a cold build cannot fit in the budget, the escalation order is documented in research R8. Its last
two steps — publishing one image per compute capability, and moving the OpenCV stage to a manually
built base image — are operator decisions, not an agent's. Both change what this contract promises:
the first rewrites the tag namespace in [container-interface.md](./container-interface.md) §1 and §9,
so taking it means updating this contract and the README alongside the workflows.

---

## 6. Immutability

- A release tag is written once. Workflows must never overwrite, move, or delete one (FR-024).
- Re-running `release.yml` for an existing tag must not move it. **This is enforced, not assumed**:
  the first job in `release.yml` inspects the registry for the pushed tag and fails the run before
  anything is built if it is already published. Nothing else provides this — GHCR accepts a second
  push to an existing tag silently, and `docker/build-push-action` has no opinion about it. Since
  this build is not bit-reproducible, an unguarded re-run would replace a pinned image with a
  different one under the same name.
- `latest` moves on release. No documentation may present it as a deployment target.
- Development tags embed both the run ID and the commit SHA, so they cannot collide with each other
  or with a release tag.

---

## 7. Release output

On a tag push, after verification passes:

- A GitHub release named for the tag.
- Release notes stating what changed and, per FR-028, explicitly calling out any breaking change: a
  removed model, a changed default, a dropped GPU generation, or a new required setting.
- The release is **not** confirmed until the maintainer has pulled it on a GPU host and run the
  runtime verification (Constitution, Development Workflow §3). CI proves the build; only hardware
  proves the runtime.
