# Model directory provenance

Everything in this directory is either a **vendored input** (committed here, in git, reviewable
in a diff) or a **pinned record** of an input the build fetches for itself. No model binary is
committed — see [`../.gitignore`](../.gitignore).

Constitution Principle II forbids a floating reference as a build input. Both text files below
are published upstream only from a branch ref, so they are vendored rather than fetched.

## Vendored files

| File | Upstream | Pinned at |
|---|---|---|
| `yolov4.cfg` | `AlexeyAB/darknet`, `cfg/yolov4.cfg` | commit `59596d7880f6504768df41d6daa586f5cb2b932f` |
| `coco.names` | `AlexeyAB/darknet`, `data/coco.names` | commit `59596d7880f6504768df41d6daa586f5cb2b932f` |

`yolov4.cfg` records its own provenance in a `#` comment at the top of the file, which the
Darknet config parser ignores.

**`coco.names` deliberately carries no such comment.** It is read as a strict one-class-per-line
list — the first line *is* class 0 — so a comment header would rename `person` to the comment
text and shift nothing else, silently mislabelling every detection. Its provenance is recorded
here instead. If you update it, update the commit SHA in this table in the same commit.

Only the YOLOv4 Darknet model needs a labels file. YOLO11 ONNX reads its class names from
embedded model metadata (`yolo_onnx.py:72-88` at pyzmNg `v2.5.1`), so no `coco.names` belongs
alongside the `.onnx` files.

## Fetched and verified at build time

`checksums.sha256` records the SHA-256 of every model binary the build downloads. The build
verifies each against this file and fails on a mismatch (FR-014) — a model that changes silently
produces detection differences indistinguishable from a code regression.

| Artifact | Source |
|---|---|
| `yolo11m.pt`, `yolo11s.pt` | `ultralytics/assets` release `v8.4.0` |
| `yolov4.weights` | `AlexeyAB/darknet` release `darknet_yolo_v3_optimal` |

The `.pt` files are inputs, not outputs: they are exported to ONNX in a throwaway build stage and
only the resulting `.onnx` reaches the shipped image. The exported `.onnx` is **not**
bit-reproducible across toolchain versions, which is why the *inputs* are gated by checksum and
the outputs are not (research R5).
