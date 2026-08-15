<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0
Rationale: MINOR. A principle is added and existing guidance is materially expanded. Nothing is
removed and no existing practice is invalidated — this repository has published no images yet, so
the registry change below breaks no promise to anyone.

Both amendments originate in explicit operator direction given while specifying
`specs/001-pyzm-serve-gpu-image`, and were flagged there as constitutional conflicts rather than
silently resolved.

Principles modified:
  II.  Pin Everything — expanded with a closing note tying it to the new Principle VI. Pinning
       governs *what* goes into the image; VI governs *when* it goes in. Neither alone is enough
       for a tag to mean anything.
  IV.  Releases Are Immutable Promises — the publication target narrows from "both Docker Hub and
       GHCR" to GHCR only, with rationale. Label and SBOM obligations are unchanged.

Principles added:
  VI.  Self-Contained by Construction — the image must contain everything it needs at publish
       time and must acquire nothing at start-up. Closes a real gap: Principle II mandated pinning
       inputs but never forbade fetching them at run time, under which the same tag could yield
       different environments on different days.

Sections modified:
  - Build & Runtime Constraints — the base image and CUDA architecture paragraphs now record where
    the deferred decisions were actually taken, so a reader does not mistake a settled choice for
    an open question. The decisions themselves remain the feature spec's, not this document's.

Sections added or removed: none.

Deferred by decision, not oversight:
  - Nothing. No TODO placeholders remain.

History: the 1.0.0 ratification report, including the corrections it recorded against an
uncommitted draft, is preserved in git history.

RATIFICATION_DATE 2026-08-15 (unchanged; date of first adoption).
-->

# docker-pyzm-serve Constitution

This repository builds one container image: `python -m pyzm.serve`, the ML inference gateway for
ZoneMinder object detection, running on a GPU. It is the successor to `jantman/docker-zm-mlapi`,
whose Dockerfile still carries the TODO this project exists to resolve — *"replace python3-opencv
with OpenCV > 4.3 with GPU support"*.

It is a **public, personal, best-effort open source project**. It is maintained by one person for
their own use, and it is explicitly intended to be usable by other people running ZoneMinder with
an NVIDIA GPU. Those two facts are in tension, and every principle below is an attempt to hold
both: build the thing the author needs, but never in a way that quietly assumes the author's
hardware, network, or habits.

## Core Principles

### I. Prove the GPU Path, Never Assume It (NON-NEGOTIABLE)

The image MUST NOT be buildable, publishable, or deployable without positive evidence that GPU
inference is actually possible.

- The Dockerfile MUST assert, in a `RUN` step that fails the build, that
  `cv2.getBuildInformation()` reports CUDA support. This works on a GPU-less CI runner because it
  proves what was *compiled*, not what is *available*.
- CI MUST independently re-verify the published artifact after the build. The Dockerfile
  assertion guards the source; the CI check guards against a bad layer cache hit, a registry
  mixup, or a tag pointing at something other than what was just built.
- Runtime GPU verification — `cv2.cuda.getCudaEnabledDeviceCount()` returning non-zero, and
  `nvidia-smi` showing utilisation during inference — requires a real GPU and therefore belongs
  to whoever deploys it. The README MUST document exactly how to check, as a copy-pasteable
  command, because a user who cannot easily verify will assume it works.
- A log line stating which processor was *requested* is NOT evidence of which was *used*. Any
  check that could pass while the GPU sits idle is not a check.

*Rationale:* The stack this replaces ran every frame on the CPU for over a year without anyone
noticing, because a PyPI `opencv-python` wheel silently shadowed a CUDA-enabled build and pyzm
logged `processor:gpu` for a request that had fallen back. That failure is worse for a stranger
than it was for the author: they have no reason to suspect it, and the symptom is merely that
detection feels slow. Shipping an image that *might* be using the GPU is the single most harmful
thing this project could do to the people who install it.

### II. Pin Everything

Every input to the image MUST be pinned to an exact, immutable identifier.

- The base image MUST be pinned to an exact tag; a digest is preferred. Never `latest`, never a
  floating major or minor.
- Upstream source checkouts (pyzmNg, and anything else built from git) MUST be pinned to a full
  commit SHA, supplied as a build `ARG` — the pattern already used by `docker-zm-mlapi`'s
  `PYZM_REF` and `MLAPI_REF`.
- Model weights MUST be pinned by version and verified by checksum. A model that silently
  changes produces detection differences indistinguishable from a code regression.
- OS package versions SHOULD be pinned where a floating version could change the CUDA or OpenCV
  toolchain. Pinning every apt package is not required and is not worth the maintenance.

*Rationale:* Reproducibility is what lets anyone — the author comparing a migration against
recorded events, or a user reporting a regression — distinguish "the image changed" from
"my setup changed". An unpinned input makes every bug report unanswerable.

This principle governs *what* goes into the image. Principle VI governs *when* it goes in — at
build time, never at start-up. Pinning an input that is fetched when the container starts still
leaves a tag whose meaning can change; both principles are required for a tag to mean anything.

### III. The Server Is a Dumb Inference Engine

The image runs a model and answers detection requests. It MUST NOT acquire any other
responsibility.

- No ZoneMinder awareness: no API client, no credentials, no event or monitor concepts.
- No orchestration: frame selection, zone filtering, nuisance filtering, confidence thresholds,
  past-detection matching and notification all belong to the client and MUST stay there.
- No configuration file. `pyzm.serve` is configured by CLI flags and by the parameters on each
  request; this repository MUST NOT reintroduce a config file, which is what made the `mlapi`
  it replaces awkward to deploy.
- The HTTP surface is upstream's (`/health`, `/models`, `/infer`, `/login`). This repository
  MUST NOT add endpoints or alter their semantics. Needing to do so is a signal to contribute
  upstream instead.

*Rationale:* This is pyzmNg's own stated design — "the server is a dumb inference engine" — and
keeping to it is what makes local and remote detection behave identically. It also keeps this
image a commodity: a user who outgrows it can swap it out, and nothing they configured elsewhere
has to change. An image that has learned about ZoneMinder is one its users cannot escape.

### IV. Releases Are Immutable Promises

A published tag MUST always refer to the same image, for as long as anything might pull it.

- Releases are cut by pushing a git tag; the image tag equals the git tag. `latest` MUST NOT be
  recommended as a deployment target.
- A published tag MUST NEVER be overwritten, moved, or deleted. Strangers pin tags in their own
  compose files and Puppet manifests, and have no way to know a tag moved under them. Deleting
  one breaks a rollback path for people the author will never hear from.
- Breaking changes — a removed model, a changed default, a new required flag — MUST be
  signalled in the release notes and in the version number. Users MUST be able to tell from the
  tag alone whether an upgrade is safe.
- Images MUST be published to **GitHub Packages (GHCR) and nowhere else**. Docker Hub MUST NOT be
  used. Published images MUST carry OCI source, revision, version and url labels, and MUST include
  an SBOM — the metadata practice inherited from `docker-zm-mlapi`.
- Builds from `main` MUST publish under a distinct, non-release tag that cannot be confused with
  a release.

*Rationale:* The author's own consumer is a live home security system whose rollback plan is
"revert and re-apply", which only works if the old tag is still pullable. Other users' setups are
invisible but no less real, and their recovery paths depend on the same promise.

Publishing to one registry rather than two follows from the same reasoning. Two registries mean two
sets of credentials, two ways for a push to half-succeed, and two places a tag can disagree with
itself — all in exchange for redundancy this project has never used. GHCR shares an account and a
permission model with the source, so provenance is one hop rather than two. This narrowing costs
nothing to existing users: this repository has never published to Docker Hub, so no promise made to
anyone is being withdrawn. The predecessor image that does live there belongs to a different
repository, and its obligation to stay pullable is unaffected by this document.

### V. Public, Personal, Best-Effort

This is a one-person project that strangers are welcome and expected to use. Both halves are
binding.

**Because it is public and meant to be used:**

- Nothing author-specific may be baked in. No hostnames, IP addresses, paths, credentials, or
  hardware assumptions from the author's network. Anything site-specific MUST be a documented
  runtime parameter with a sensible general default.
- The README MUST be sufficient for someone who has never seen this repository to run the image
  against their own ZoneMinder and confirm the GPU is working. A `docker-compose.yml` and a
  worked example are the baseline, matching `docker-zm-mlapi`.
- Defaults MUST be reasonable for a typical user, not tuned to the author's hardware.

**Because it is personal and best-effort:**

- The README MUST carry a repostatus badge and state plainly that support is best-effort, that
  issues may not be addressed, and that PRs are welcome but not guaranteed a review — the same
  honest framing `docker-zm-mlapi` uses. Setting expectations is a feature.
- There is no obligation to support hardware, operating systems, or use cases the author cannot
  test. Declining is always acceptable; pretending is not.
- YAGNI still applies. Generality is added when a real user needs it, not in anticipation.

*Rationale:* The failure mode for a project like this is not abandonment, which users can cope
with — it is a project that looks supported and is not, or one that appears general and is
quietly hardcoded to one person's basement. Saying plainly what this is costs nothing and
prevents both.

### VI. Self-Contained by Construction

The published image MUST contain everything it needs to serve inference, and MUST acquire no part
of its own runtime after publication.

- Models, libraries and the gateway itself MUST be baked in at build time. No download on first
  start, no model directory the image cannot run without, no setup step deferred to the operator.
- Model artifacts MUST be checksum-verified **at build time**, so a changed artifact fails the
  build. A user MUST NOT be the one to discover that a model changed.
- The image MUST start, become healthy, and answer an inference request with no network access and
  no volumes mounted. This is the test of this principle, and it is cheap enough to run every time.
- Two containers started from the same tag MUST have identical environments — same models, same
  versions, same checksums — regardless of when or where they start. Any behavioural difference
  between them MUST be attributable to explicit operator configuration, never to what the image
  fetched at start-up.
- Operators MAY supply additional models of their own. Adding to the shipped set is a feature;
  needing to supply something before the image works at all is a violation of this principle.

*Rationale:* A tag that fetches something when it starts is not a version, it is a recipe — and its
ingredients can change without the tag changing, which empties the promise made in Principle IV.
Worse, run-time fetches fail precisely when they are least affordable: during a recovery, on a
network that is down, behind a firewall added since deployment, or against an upstream URL that has
since disappeared. An inference gateway for a security camera system is restarted in exactly those
circumstances. Baking everything in costs image size, which is paid once at pull time by a machine,
rather than availability, which is paid at the worst possible moment by a person.

## Build & Runtime Constraints

**OpenCV.** Built from source with `WITH_CUDA=ON`. Debian and Ubuntu both ship OpenCV without
CUDA, which is why building from source is not optional.

**CUDA architectures.** The image MUST be built for **several CUDA compute capabilities**, not
only the author's. Users have different GPUs, and an image that only works on one is not usable
by anyone else — which Principle V forbids. The `CUDA_ARCH_BIN` list MUST be an explicit build
`ARG`, MUST cover a reasonable spread of currently-supported NVIDIA generations, and MUST include
`6.1` (Pascal) for as long as it is supported. Note that Pascal has a finite life: NVIDIA's 580
branch is the last to support it, and Debian 14 will ship 590+.

This costs a materially longer build and a larger binary. That is accepted deliberately as the
price of the image being usable by anyone other than its author.

*The current list is set by `specs/001-pyzm-serve-gpu-image` at 6.1, 7.5, 8.6 and 8.9 — Pascal,
Turing, consumer Ampere and Ada — plus forward-compatible intermediate code for the newest, so a
card newer than any built for still runs. Adjusting that list is a normal change to that spec, not
an amendment to this document; what this document fixes is the shape of the constraint, not the
values.*

**Install order is load-bearing.** Ultralytics and several other packages pull `opencv-python`
from PyPI, which shadows a source-built OpenCV. Whatever order resolves this MUST be commented
in the Dockerfile explaining *why*, because the failure it prevents is silent and the fix looks
arbitrary to anyone who has not been bitten by it. A future cleanup that "tidies" an
unexplained workaround would restore a year-long bug. Principle I's assertion is the backstop,
not the solution.

**Base image.** Deliberately not fixed by this constitution. NVIDIA does not publish
Debian-based CUDA images, so matching `docker-zm-mlapi`'s Debian convention requires adding
NVIDIA's CUDA apt repository, while an `nvidia/cuda` Ubuntu tag gets a known-good toolchain at
the cost of diverging from the sibling repository. Either is acceptable; the choice belongs to
the first feature spec. Whatever is chosen MUST be pinned per Principle II.

*That choice has since been taken: `specs/001-pyzm-serve-gpu-image` selects NVIDIA-published CUDA
images of the 12.4 generation, pinned by digest, using the development variant to build and the
runtime variant to ship. Recorded here so the deferral above is not misread as still open. It
remains the spec's decision to revisit.*

**Build cost.** Compiling OpenCV with CUDA for multiple architectures is the dominant cost of
every build. Layers MUST be ordered so that this stage caches and is not invalidated by changes
to application code, models, or metadata. A multi-stage build that ships only the runtime
artifacts is strongly preferred — users should not download a compiler toolchain.

**Platforms.** `linux/amd64` is the supported platform today. This is a statement of what is
tested, not a principle; adding a platform the author can test is a normal change.

**Models.** `yolo11m.onnx` is the primary model and `yolo11s.onnx` a lighter alternative; YOLOv4
Darknet weights are retained as a fallback known to work with OpenCV DNN's CUDA backend. Which
models ship MUST be documented, since it determines what a user can request without supplying
their own. All of them ship inside the image, per Principle VI.

**Healthcheck.** The image MUST define a `HEALTHCHECK` against `/health`, so a wedged gateway is
visible to Docker rather than only as detection silently stopping.

## Development Workflow

Deliberately minimal. Process is added when it earns its place.

1. **Work on `main`.** Commits land directly on the default branch. Feature branches and pull
   requests are not required for the maintainer's own work and MUST NOT be introduced on an
   agent's own initiative. Contributor PRs are a separate matter and are welcome.
2. **Green before done.** The image must build — which, per Principle I, means the CUDA
   assertion passed — and CI must be green. A build that cannot publish is not finished work.
3. **Verify on real hardware before releasing.** CI proves the build; only a machine with a GPU
   proves the runtime. A release is not confirmed until it has been pulled and checked there.
4. **Integration checks are useful but not authoritative.** `contracts/container-images.md` in
   the author's `privatepuppet` repository describes what one consumer expects, and is a good
   real-world test. It does not define this project's scope, and this project MUST NOT acquire
   requirements that only make sense for that deployment.
5. **Stop when unclear.** On genuine confusion or an unplanned significant decision, stop and
   ask the operator. Do not guess.

Commit messages open with a concise one-sentence summary followed by a detailed explanation of
the change and its reasoning.

## Governance

**Authority.** This constitution supersedes other conventions in this repository where they
conflict. It governs this repository only.

**Amendment procedure.** Amendments are made by editing `.specify/memory/constitution.md` in a
commit stating the reason, the version bump, and any follow-up work created. The repository
maintainer approves all amendments.

**Versioning policy.** Semantic versioning applies to this document:
- MAJOR — a principle is removed or redefined in a way that invalidates existing practice.
- MINOR — a principle or section is added, or existing guidance is materially expanded.
- PATCH — clarification, wording, or typo fixes that do not change what is required.

Note that this is the versioning of *this document*, and is unrelated to the image's release
tags, which follow Principle IV.

**Compliance review.** Compliance is checked before each commit and before each release. A
change that violates a principle is either corrected or accompanied by an explicit written
justification in the commit message. Repeated justification of the same violation is a signal to
amend this constitution rather than keep granting exceptions.

**Version**: 1.1.0 | **Ratified**: 2026-08-15 | **Last Amended**: 2026-08-15
