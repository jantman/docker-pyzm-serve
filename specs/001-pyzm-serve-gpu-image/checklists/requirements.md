# Specification Quality Checklist: GPU-Accelerated pyzm.serve Inference Gateway Image

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-08-15
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

### Resolved clarifications (2026-08-15)

All three open decisions were answered by the operator and written into the spec. No markers remain.

- **Models — baked in at build time.** FR-013 and FR-014 now require all three model artifacts to be
  present in the published image, checksum-verified at build, with no run-time fetch. The operator
  additionally generalised this into a rule about the image as a whole, captured as new requirements
  FR-033 and FR-034 and as SC-013 and SC-014.
- **Base image — NVIDIA's own CUDA images.** Recorded as a settled decision in Assumptions: 12.4
  generation, pinned by digest, development variant for building and runtime variant for the
  published stage. Exact tags belong to `/speckit-plan`.
- **GPU generations — Pascal, Turing, consumer Ampere, Ada** (6.1, 7.5, 8.6, 8.9) plus
  forward-compatible intermediate code for the newest. FR-017 states this; the edge-case list now
  distinguishes a too-new GPU (runs, with a first-load delay) from a too-old one (does not run).

### Judgement calls recorded during validation

- **"No implementation details"** is marked satisfied despite the spec naming GPUs, CUDA, container
  healthchecks, git tags and GitHub Packages. For a container-image product these are the deliverable
  itself and the user's own vocabulary, not a technology chosen to realise some more abstract
  requirement. The spec does avoid genuine implementation choices: no base image, no build-stage
  layout, no workflow file structure, no specific commands or flags. Those belong to `/speckit-plan`.
- **Success criteria** are stated as observable outcomes (latency at a percentile, absence of a
  fallback signature in logs, time-to-first-success for a newcomer, percentage of releases verified)
  rather than as internal mechanics. SC-001's 300 ms figure is anchored to the ~800 ms per-frame CPU
  baseline recorded for the system this image replaces.

### Constitutional amendments — raised here, landed in v1.1.0 (2026-08-15)

Two conflicts were flagged during validation rather than silently resolved. Both were carried by a
single `/speckit-constitution` run and are now closed; this feature no longer contradicts the
Constitution, and a compliance review should come back clean.

1. **Registry.** FR-021 (GHCR only) contradicted v1.0.0 Principle IV's "Images MUST be pushed to
   both Docker Hub and GHCR". Resolved by narrowing the principle's publication target. Treated as
   MINOR, not MAJOR: the principle's substance — a published tag never moves — is untouched, and
   since this repository has never published to Docker Hub, no promise to any user was withdrawn.
2. **Self-containment.** FR-033 and FR-034 state that the image must be self-contained and that the
   same tag must always yield the same environment, models included. Principle II ("Pin Everything")
   mandated pinning inputs but did not forbid acquiring them at run time — the gap these requirements
   close. Raised into the Constitution as **Principle VI, "Self-Contained by Construction"**, since it
   constrains every future change to the image, not just this feature.

The Constitution's Build & Runtime Constraints section now also records that the base-image and
CUDA-architecture decisions it had deferred were taken in this spec, so neither reads as still open.
The values remain this spec's to revisit.

### Lifecycle

- Items marked incomplete require spec updates before `/speckit-clarify` or `/speckit-plan`
