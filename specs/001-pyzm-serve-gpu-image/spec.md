# Feature Specification: GPU-Accelerated pyzm.serve Inference Gateway Image

**Feature Branch**: `main` *(per Constitution "Development Workflow" §1, work lands directly on the default branch; no feature branch is created)*

**Created**: 2026-08-15

**Status**: Draft

**Input**: User description: "We need to implement this project from scratch including CI/CD. See `~/GIT/privatepuppet/specs/001-zm-es7-migration/contracts/container-images.md` for information on the image this project should provide (Image B - pyzm.serve) and look at `~/GIT/docker-zm-mlapi` for an example of how we handle Docker image creation and CI/CD (note that we now only want to push to GitHub Packages / GHCR; we no longer use Docker Hub)."

---

## Overview

This repository currently contains nothing but a licence and a placeholder README. This feature
builds the entire project: a published container image that runs the `pyzm.serve` object-detection
gateway on an NVIDIA GPU, plus the documentation and automated pipeline that publish and verify it.

It is the successor to `jantman/docker-zm-mlapi`, whose vision library was built without GPU
support — the defect that had every frame of a live security system processed on CPU, at roughly
800 ms per frame, for over a year without anyone noticing. Everything below is shaped by the fact
that this failure is *silent*: a gateway that has quietly fallen back to CPU still answers every
request correctly, just slowly. The single most important property of this feature is that such a
fallback cannot be shipped undetected.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run object detection on the GPU (Priority: P1)

Someone running ZoneMinder with an NVIDIA GPU pulls a published tag, starts the container with GPU
access, and points their detection client at it. Frames posted to the gateway come back with
labelled bounding boxes, computed on the GPU, materially faster than the CPU-bound gateway this
image replaces. When something is wrong with the GPU path, they find out immediately and loudly —
not months later from a vague sense that detection feels sluggish.

**Why this priority**: This is the product. Every other story exists to deliver, verify, or explain
this one. A working image with no CI is still useful to its author; CI with no working image is
worth nothing.

**Independent Test**: Build the image locally, run it with GPU access on a host with a supported
NVIDIA card, post a sample frame to the inference endpoint, and confirm detections come back while
`nvidia-smi` shows GPU utilisation and the logs contain no CPU-fallback message.

**Acceptance Scenarios**:

1. **Given** a host with a supported NVIDIA GPU and working container GPU support, **When** the
   operator starts the image with GPU access and no extra configuration, **Then** the gateway
   becomes healthy and reports the primary object-detection model as loaded.
2. **Given** a running, healthy gateway, **When** a client posts an image frame requesting object
   detection, **Then** the response contains zero or more detections, each with a label, a
   confidence, a bounding box, and the name of the model that produced it, and reports no error.
3. **Given** a running, healthy gateway on a GPU host, **When** the operator runs the verification
   command from the README, **Then** it reports a non-zero count of CUDA-capable devices and the
   logs contain no message indicating a fallback to CPU or a vision library lacking CUDA support.
4. **Given** a running gateway, **When** the operator inspects the model listing, **Then** it names
   exactly the models the README says the image ships, so the operator can confirm the image is
   what they think it is without reading its source.
5. **Given** a gateway whose process has wedged and stopped answering, **When** the container
   runtime evaluates the healthcheck, **Then** the container is reported unhealthy rather than
   appearing to run normally.
6. **Given** a host with no GPU, or a container started without GPU access, **When** the gateway is
   asked to serve GPU inference, **Then** the failure is stated explicitly in the logs and in the
   health state — the gateway MUST NOT silently serve CPU inference as if nothing were wrong.

---

### User Story 2 - Cut an immutable, verified release (Priority: P2)

The maintainer pushes a git tag. Unattended, the project builds the image, publishes it to GitHub
Packages under a tag identical to the git tag, independently re-verifies that the *published*
artifact really was compiled with GPU support, and creates a GitHub release. Anyone — including the
maintainer's own Puppet-managed home security system — can pin that tag forever and trust it will
never change under them.

**Why this priority**: Without this, the image cannot be consumed by anything but the machine that
built it. It ranks below the image itself only because an image that works can be published by hand
once, whereas a pipeline that publishes a broken image is worse than no pipeline.

**Independent Test**: Push a throwaway pre-release tag and confirm that a correspondingly tagged
image appears in GitHub Packages, that the post-publish verification step ran and passed against
the pulled artifact, that a GitHub release was created, and that the image carries provenance
metadata identifying the exact source revision it was built from.

**Acceptance Scenarios**:

1. **Given** a commit whose image builds successfully, **When** a git tag is pushed, **Then** an
   image tagged identically to the git tag is published to GitHub Packages and nowhere else.
2. **Given** a published release image, **When** the pipeline pulls that image back from the
   registry and re-runs the GPU-support assertion, **Then** the assertion passes; if it fails, the
   pipeline fails visibly and the release is not presented as usable.
3. **Given** a published release image, **When** anyone inspects its metadata, **Then** it carries
   the source repository, the exact source revision, and the version it was built as, together with
   a software bill of materials.
4. **Given** a release tag that has already been published, **When** any workflow runs again,
   **Then** that tag is not overwritten, moved, or deleted.
5. **Given** a user reading the documentation, **When** they look for a tag to deploy, **Then** they
   are directed to an immutable version tag and never to a mutable rolling tag.

---

### User Story 3 - Get fast feedback on every change (Priority: P3)

The maintainer commits to the default branch. The project builds the image automatically, fails the
build outright if the vision library in the image was not compiled with GPU support, and publishes
the result under a tag that could never be mistaken for a release, so it can be pulled onto the GPU
host and tried.

**Why this priority**: Compiling a CUDA-enabled vision library for several GPU generations is slow
and easy to get subtly wrong. Discovering a mistake at release time, after a long build, is
expensive. This story exists to move that discovery earlier — but the project is usable without it.

**Independent Test**: Push a commit to the default branch and confirm a non-release-tagged image
appears in GitHub Packages; then push a commit that deliberately breaks the GPU build path and
confirm the pipeline fails rather than publishing.

**Acceptance Scenarios**:

1. **Given** a commit pushed to the default branch, **When** the pipeline runs, **Then** an image is
   published under a tag that is unambiguously not a release and cannot collide with one.
2. **Given** a change that causes the image's vision library to lose GPU support — for example a
   dependency that reintroduces a pre-built CPU-only vision package that shadows the source-built
   one — **When** the pipeline runs, **Then** the build fails and nothing is published.
3. **Given** the pipeline runs on a build machine with no GPU, **When** the GPU-support assertion
   executes, **Then** it still produces a correct verdict, because it tests what was *compiled*, not
   what hardware happens to be present.
4. **Given** the maintainer wants to rebuild without a code change, **When** they trigger the
   pipeline manually, **Then** it runs and publishes exactly as it would on a push.

---

### User Story 4 - Adopt the image as a stranger (Priority: P4)

Someone who has never seen this repository, running ZoneMinder with an NVIDIA GPU, finds it, reads
the README, and gets a working, GPU-verified gateway on their own hardware. They can tell at a
glance that this is a personal, best-effort project, they can tell which models it ships, and they
can tell whether their GPU is one it was built for — before they spend an evening on it.

**Why this priority**: Constitution Principle V makes this binding, not optional. It ranks last
only because it can be completed after the image and pipeline exist, and it changes nothing about
how they work.

**Independent Test**: Hand the README to someone with a GPU host and no knowledge of the project,
and have them reach a verified-working gateway using only copy-pasteable commands from it, without
asking a question.

**Acceptance Scenarios**:

1. **Given** a newcomer with a supported GPU, **When** they follow the README from top to bottom,
   **Then** they reach a running gateway and a command whose output proves to them that the GPU is
   actually being used.
2. **Given** a newcomer evaluating the project, **When** they open the README, **Then** they see
   plainly that this is a personal, best-effort project, that issues may not be addressed, and that
   PRs are welcome but not guaranteed a review.
3. **Given** a newcomer with an unusual GPU, **When** they read the README, **Then** they can tell
   which GPU generations the published image was built for and what to do if theirs is not among
   them.
4. **Given** a newcomer who wants to try it before wiring it into ZoneMinder, **When** they use the
   supplied worked example, **Then** the gateway starts with GPU access and answers a sample
   inference request.
5. **Given** the repository is public, **When** anyone reads any part of it, **Then** they find no
   hostname, address, path, credential, or hardware assumption specific to the maintainer's network.

---

### Edge Cases

- **The gateway is started without GPU access** (the `--gpus` flag forgotten, or no driver on the
  host). The gateway must make this loud — in the logs and in the health state — rather than serving
  CPU inference that merely feels slow. This is the exact failure the project exists to prevent.
- **A dependency reintroduces a CPU-only vision library** that shadows the source-built one. The
  build must fail. The countermeasure must be commented in place explaining *why*, so a future
  tidy-up does not delete an unexplained workaround and restore a year-long silent bug.
- **A stale build-cache layer** yields an image different from what the source describes. The
  post-publish verification runs against the artifact actually pulled from the registry, not against
  anything the build step reports about itself.
- **The build exceeds the hosted runner's job time limit.** Compiling a CUDA-enabled vision library
  for several GPU generations is the dominant cost of every build and scales with the number of
  generations. If the build cannot finish inside the limit, that is a scope decision to be taken
  deliberately (fewer generations, or different build infrastructure), not a flaky pipeline to be
  retried.
- **A user's GPU is newer than any generation the image was built for.** The forward-compatible
  intermediate code required by FR-017 must let it run, at the cost of a noticeable one-off delay on
  first load. That delay must be documented so it is not mistaken for a hang.
- **A user's GPU is older than any generation the image was built for.** Inference will not run at
  all, and no forward-compatibility mechanism helps. The README must say which generations are
  covered so this is discoverable before deployment, not after.
- **A user's GPU has too little memory for the primary model.** The lighter alternative model must be
  selectable at run time without rebuilding the image.
- **A client requests a model the gateway did not load.** The gateway returns no detections and an
  explanatory error; it does not crash, and the client is expected to log and continue.
- **A model artifact is fetched and does not match its recorded checksum.** The build fails. A model
  that changes silently produces detection differences indistinguishable from a code regression.
- **An upstream endpoint's behaviour changes.** Because this image adds no endpoints and alters no
  semantics, such a change is upstream's and must be surfaced as a version bump, not papered over
  locally.
- **A release tag is pushed that already exists**, or a workflow is re-run for an existing tag. The
  published tag must not move. Someone, somewhere, has pinned it.
- **The Pascal generation reaches end of life.** Support for it is required for as long as it is
  supported upstream; its eventual removal is a breaking change that must be signalled in the
  version number and the release notes.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Inference gateway behaviour

- **FR-001**: The image MUST run the upstream `pyzm.serve` object-detection gateway as its main
  process, listening on a network port, and MUST NOT require or read a configuration file.
- **FR-002**: The image MUST expose the upstream HTTP surface — health, model listing, inference,
  and login — unchanged. It MUST NOT add endpoints, remove them, or alter their semantics.
- **FR-003**: Operators MUST be able to select, at container start and without rebuilding, at least:
  which models are loaded, whether inference runs on GPU or CPU, and the listening port.
- **FR-004**: The gateway MUST perform inference on the GPU when GPU inference is requested and a
  supported GPU is available, and MUST report an explicit, visible failure when GPU inference is
  requested and cannot be provided. It MUST NOT silently substitute CPU inference for requested GPU
  inference.
- **FR-005**: The image MUST define a container healthcheck against the gateway's health endpoint,
  so that a wedged gateway is visible to the container runtime rather than manifesting only as
  detection quietly stopping.
- **FR-006**: The image MUST NOT contain any ZoneMinder-specific knowledge: no ZoneMinder API
  client, no ZoneMinder credentials, and no awareness of events, monitors, or zones. Frame
  selection, zone filtering, nuisance filtering, confidence policy and notification remain entirely
  the client's responsibility.
- **FR-007**: Request authentication MUST be available as an opt-in run-time setting and MUST be off
  by default, with the security implication of the default stated in the documentation.

#### Proving the GPU path

- **FR-008**: The build MUST fail if the vision library that the final image will actually load was
  not compiled with CUDA support. This check MUST be valid on a build machine with no GPU, because
  it tests what was compiled rather than what is available.
- **FR-009**: The build MUST fail if a pre-built CPU-only vision library takes precedence over the
  source-built one in the final image, regardless of how it was introduced. The mechanism that
  prevents this MUST be documented in place, in the build definition, with an explanation of the
  silent failure it prevents.
- **FR-010**: After publishing, the pipeline MUST independently re-verify the published artifact by
  retrieving it from the registry and re-running the compiled-with-CUDA assertion against it. A
  failure here MUST fail the pipeline visibly.
- **FR-011**: A log line stating which processor was *requested* MUST NOT be treated as evidence of
  which was *used*. No verification may pass under conditions where the GPU is idle.
- **FR-012**: The documentation MUST give a copy-pasteable command that an operator runs on their own
  GPU host to confirm a non-zero count of CUDA-capable devices, and a second that lets them observe
  GPU utilisation during a real inference request.

#### Models

- **FR-013**: The image MUST make available a primary object-detection model, a lighter alternative
  model that is present but not loaded by default, and a legacy fallback model that is present but
  disabled. All three MUST be baked into the published image at build time. The image MUST NOT fetch
  a model at run time, and MUST serve every model it ships with no network access and no writable
  model volume.
- **FR-014**: Every model artifact MUST be pinned to an exact version and verified against a recorded
  checksum **at build time**. A checksum mismatch MUST fail the build, never warn. A user must never
  be the one to discover that a model changed.
- **FR-015**: The documentation MUST state exactly which models the image ships and how an operator
  supplies a model of their own. Operator-supplied models are an addition to the shipped set, never a
  precondition for the image working.

#### Reproducibility

- **FR-016**: Every external input to the image MUST be pinned to an exact, immutable identifier: the
  base image to an exact tag or digest, every upstream source checkout to a full commit revision
  supplied as a build argument, and every model artifact to a version plus checksum. No floating
  references — no rolling tags, no branch names, no "latest".
- **FR-017**: The set of GPU compute capabilities the image is built for MUST be an explicit,
  overridable build argument, and MUST cover the Pascal, Turing, consumer Ampere and Ada generations
  (compute capabilities 6.1, 7.5, 8.6 and 8.9). The image MUST additionally carry forward-compatible
  intermediate code for the newest covered generation, so that a card newer than any built for still
  runs, at the cost of a one-off compilation delay on first load. Pascal MUST remain covered for as
  long as it is supported upstream.
- **FR-018**: The published image MUST NOT contain a compiler toolchain or other build-only
  artifacts. Users must not have to download a build environment in order to run inference.
- **FR-019**: The build MUST be arranged so that the expensive GPU vision-library compilation is not
  invalidated by changes to application code, models, documentation, or metadata.
- **FR-020**: The published image MUST support the `linux/amd64` platform. Other platforms are out of
  scope for this feature.

#### Publishing and release

- **FR-021**: Images MUST be published to GitHub Packages (GHCR) only. Docker Hub MUST NOT be used.
  *(Constitution v1.1.0, Principle IV.)*
- **FR-022**: Every push to the default branch, and every manual trigger, MUST build and publish an
  image under a tag that is unambiguously not a release and cannot collide with a release tag.
- **FR-023**: Pushing a git tag MUST build and publish an image whose registry tag is identical to
  the git tag, and MUST create a corresponding GitHub release.
- **FR-024**: A published tag MUST NEVER be overwritten, moved, or deleted. If the release process
  also maintains a mutable rolling tag, the documentation MUST NOT present that tag as a deployment
  target.
- **FR-025**: Every published image MUST carry provenance metadata identifying the source repository,
  the exact source revision it was built from, and the version it was built as, and MUST include a
  software bill of materials.
- **FR-026**: The pipeline MUST NOT present an image as a usable release if any verification step for
  that image did not run or did not pass.
- **FR-027**: The pipeline MUST run to completion on standard GPU-less hosted build machines. Runtime
  GPU verification is explicitly the deploying operator's responsibility, not the pipeline's.
- **FR-028**: Breaking changes — a removed model, a changed default, a dropped GPU generation, a new
  required setting — MUST be signalled in the release notes and in the version number, so that a
  user can judge from the tag alone whether an upgrade is safe.

#### Documentation and project hygiene

- **FR-029**: The README MUST be sufficient for someone who has never seen the repository to run the
  image against their own ZoneMinder and confirm the GPU is working, and MUST cover: prerequisites,
  a quickstart, the GPU verification commands, the models shipped, the GPU generations supported,
  the full run-time configuration surface, and how to pin a version.
- **FR-030**: The README MUST carry a project-status badge and state plainly that this is a personal,
  best-effort project, that issues may not be addressed, and that PRs are welcome but not guaranteed
  a review.
- **FR-031**: The repository MUST ship a worked, runnable example that starts the gateway with GPU
  access and demonstrates a successful inference request.
- **FR-032**: Nothing author-specific may appear anywhere in the repository or image — no hostnames,
  addresses, paths, credentials, or assumptions about the maintainer's hardware or network. Anything
  site-specific MUST be a documented run-time parameter with a sensible general default.

#### Self-containment

- **FR-033**: The published image MUST be self-contained. Everything needed to serve inference —
  models, libraries, and the gateway itself — MUST be present in the image at publish time. The image
  MUST NOT download, generate, or otherwise acquire any part of its own runtime after publication.
- **FR-034**: Two containers started from the same image tag MUST have identical environments,
  including identical model artifacts, regardless of when or where they are started, and regardless
  of whether they have network access. Any behavioural difference between them MUST be attributable
  to explicit operator configuration, never to what the image fetched at start-up.

---

### Key Entities

- **Published image**: One immutable artifact per publish, identified by a registry tag, carrying
  provenance metadata and a bill of materials. Relates to exactly one source revision and one set of
  pinned inputs.
- **Release tag**: A git tag and the identically-named registry tag it produces. Immutable once
  published; the unit users pin and roll back to.
- **Model artifact**: A named detection model, pinned by version and checksum and baked into the
  image at build time, in one of three roles — primary (loaded by default), alternative (present, not
  loaded), legacy fallback (present, disabled).
- **Pinned input**: Any external thing the image is built from — base image, upstream source
  checkout, model artifact — each carrying an exact immutable identifier.
- **Detection**: The unit of the gateway's output — a label, a confidence, a bounding box, the
  detection type, and the model that produced it.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a host with a supported NVIDIA GPU, a single-frame object-detection request against
  the primary model completes in no more than 300 ms at the 95th percentile — at least a two-fold
  improvement on the roughly 800 ms per frame the CPU-bound gateway this image replaces takes on the
  same class of hardware.
- **SC-002**: Across a sustained run of at least 100 consecutive inference requests on a GPU host,
  zero log entries indicate a fallback to CPU or a vision library lacking CUDA support, and GPU
  utilisation is observably non-zero throughout.
- **SC-003**: An image whose vision library lacks GPU support cannot be published: in a deliberate
  test, a change that removes GPU support causes the pipeline to fail before publishing, 100% of the
  time.
- **SC-004**: 100% of published release images have been retrieved from the registry and
  re-verified after publication, with the result recorded in the pipeline run.
- **SC-005**: A person who has never seen the repository reaches a running, GPU-verified gateway on
  their own hardware in under 30 minutes, using only copy-pasteable commands from the README and
  asking no questions.
- **SC-006**: 100% of external inputs to the image resolve to an exact immutable identifier; a review
  of the build definition finds zero floating references.
- **SC-007**: Rebuilding the same source revision twice produces images that load the same model
  versions and report the same vision-library build configuration.
- **SC-008**: The published image contains no compiler toolchain, and its download size is stated in
  the README so a user knows what they are pulling before they pull it.
- **SC-009**: An unattended run from git-tag push to a pullable, verified release image completes
  within the hosted build machine's job time limit, with no manual intervention and no retries.
- **SC-010**: A client written only against the upstream gateway's documented interface works against
  this image without modification — zero endpoints added, removed, or changed.
- **SC-011**: An operator can determine which models the image serves and which GPU generations it
  supports from the running image and the README alone, without reading the build definition or the
  source.
- **SC-012**: Zero author-specific values appear in the repository or the published image, confirmed
  by review before each release.
- **SC-013**: The image starts, becomes healthy, and answers an inference request with no network
  access whatsoever, and with no volume mounted — demonstrating that nothing it needs is fetched
  after publication.
- **SC-014**: Two containers started from the same tag, on different hosts and at different times,
  report identical loaded models with identical checksums.

---

## Assumptions

- **GHCR only — settled in the Constitution.** Images are published to GitHub Packages and nowhere
  else. This originally conflicted with Constitution v1.0.0 Principle IV ("Images MUST be pushed to
  both Docker Hub and GHCR"); the conflict was flagged rather than silently resolved, and closed by
  the v1.1.0 amendment, which narrows the publication target and records why doing so withdraws no
  promise (this repository never published to Docker Hub). FR-021 now follows the Constitution rather
  than contradicting it.
- **Self-containment is a principle, not just a requirement — now in the Constitution.** FR-033 and
  FR-034 exist because the operator stated the rule directly: the image must be self-contained, and
  running the same image tag must always produce the same environment, models included, so everything
  is baked in. That is broader than this feature — it constrains every future change to the image —
  so it was raised into Constitution v1.1.0 as Principle VI, "Self-Contained by Construction",
  closing the gap left by Principle II, which mandated pinning inputs without forbidding run-time
  acquisition. FR-033, FR-034, SC-013 and SC-014 are this feature's expression of that principle.
- **Sibling repository as pattern, not as template.** `jantman/docker-zm-mlapi` is the reference for
  how this project builds and publishes images — pinned build arguments, provenance labels, bill of
  materials, build-on-push and release-on-tag. Its actual content is not reused: it builds a
  different, archived server, and its Dockerfile carries the very unresolved GPU TODO that this
  project exists to close.
- **The privatepuppet contract is an integration check, not the acceptance surface.** Per
  Constitution "Development Workflow" §4, `contracts/container-images.md` describes what one consumer
  expects and is a good real-world test. Requirements that make sense only for that deployment — a
  specific data path layout, a specific inference client — are deliberately not adopted here.
- **Upstream provides what this image serves.** `pyzm.serve` supplies the health, model-listing,
  inference and login endpoints and their behaviour; this project packages them. Endpoint semantics
  are upstream's to define and change.
- **No GPU in CI.** Standard hosted build machines have no GPU, so the pipeline can only ever prove
  what was *compiled*. Proving what is *executed* requires real hardware and belongs to the deploying
  operator, per Constitution Principle I. A release is not considered confirmed until it has been
  checked on a GPU host.
- **Authentication off by default.** The gateway is assumed to run on a trusted network segment
  alongside its client, as the consuming deployment does. Authentication is available but opt-in, and
  the default is documented rather than assumed safe everywhere.
- **`linux/amd64` only.** This states what the maintainer can test, not a principle. Adding a
  platform later is a normal change.
- **Rolling-tag policy.** A mutable convenience tag may exist, but no documentation will present it
  as a deployment target; immutable version tags are the only recommended way to deploy.
- **Base image family: settled — NVIDIA's own CUDA images.** The Constitution deferred this decision
  to the first feature spec; it is taken here. The image is built on NVIDIA-published CUDA base
  images of the 12.4 generation (matching the driver on the consuming host), pinned by digest, using
  the development variant to compile and the runtime variant for the published stage. This buys a
  toolchain NVIDIA tests, and the two-stage split is what keeps the compiler out of the shipped image
  per FR-018. It diverges from `docker-zm-mlapi`'s Debian convention; that cost is accepted in
  exchange for removing the NVIDIA apt repository as a second pinning surface and a second silent
  failure mode. The exact tags belong to `/speckit-plan`; that they are pinned by digest does not.

---

## Out of Scope

- Publishing to Docker Hub or any registry other than GitHub Packages.
- Runtime GPU verification inside the pipeline, or a self-hosted GPU build machine.
- Platforms other than `linux/amd64`.
- Any change to the ZoneMinder side of the system: the event server, the detection hook, its
  configuration, or the Puppet code that deploys them.
- Training, fine-tuning, or exporting models. This project packages published model artifacts.
- Adding to or altering the upstream gateway's HTTP interface. A need to do so is a signal to
  contribute upstream.
- Client-side concerns the Constitution places outside this image: frame selection, zone filtering,
  nuisance filtering, confidence policy, past-detection matching, notification, and metrics about
  any of them.
