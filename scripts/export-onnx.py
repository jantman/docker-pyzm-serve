#!/usr/bin/env python3
"""Export Ultralytics YOLO11 `.pt` weights to ONNX for OpenCV DNN.

Build-stage only. This runs in the throwaway `model-export` stage on a slim Python base --
NOT on the CUDA base and NOT in the shipped image. That isolation is deliberate:
`ultralytics` depends on `opencv-python`, the PyPI wheel that shadows a source-built CUDA
OpenCV and caused the year-long CPU regression this project exists to prevent. Confining it
to a stage whose site-packages never reach the runtime image removes the hazard by
construction rather than working around it (research R4).

Export parameters are not free choices -- each one matches something upstream assumes:

  imgsz=640      `yolo_onnx.py:25` sets `_DEFAULT_DIM = 640` and the backend letterboxes to
                 it unconditionally. A model exported at any other size is silently
                 mis-scaled on every frame: no error, just worse detections (MA-3).

  dynamic=False  A static input shape is what OpenCV DNN handles best, and the server only
                 ever feeds it 640x640.

  simplify=True  Folds constants and removes no-op nodes, which OpenCV's ONNX importer is
                 happier with than the raw graph.

  opset=17       Comfortably supported by OpenCV 4.12's ONNX importer.

  nms=False      The default, and left alone on purpose. `yolo_onnx.py:90-147` detects
                 end-to-end (NMS-baked) ONNX and carries a separate pre-NMS fallback path
                 for when "OpenCV produces garbled output" -- a code path whose existence is
                 a warning. The plain export takes upstream's well-trodden branch (MA-4).

Class labels need no accompanying file: Ultralytics embeds `names` in the ONNX metadata on
export, and `populate_class_labels()` (`yolo_onnx.py:72-88`) reads them from there. Only the
YOLOv4 Darknet model needs a `coco.names`.

Usage:
    export-onnx.py MODEL.pt [MODEL.pt ...] [--output-dir DIR]
"""

import argparse
import os
import shutil
import sys

# Ultralytics will pip-install missing export dependencies on demand. That would pull
# unpinned packages into the build at an arbitrary version, which Constitution Principle II
# forbids -- so it is switched off here and `onnx`, `onnxslim` and `onnxruntime` are pinned
# explicitly in the Dockerfile instead. If the export fails complaining about a missing
# package, pin that package in the Dockerfile; do not re-enable autoinstall.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
# Keep Ultralytics' settings/cache inside the build stage rather than probing for a home
# directory that may not exist.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

IMGSZ = 640
OPSET = 17


def export_one(pt_path, output_dir):
    """Export a single .pt to .onnx, returning the path written."""
    from ultralytics import YOLO

    print("=" * 78)
    print("Exporting {} -> ONNX (imgsz={}, opset={}, dynamic=False, simplify=True, "
          "nms=False)".format(pt_path, IMGSZ, OPSET))
    print("=" * 78)

    model = YOLO(pt_path)
    produced = model.export(
        format="onnx",
        imgsz=IMGSZ,
        dynamic=False,
        simplify=True,
        opset=OPSET,
        # nms is left at its default of False on purpose -- see the module docstring.
    )

    # Ultralytics returns the path it wrote, next to the source .pt.
    produced = str(produced)
    if not os.path.isfile(produced):
        raise SystemExit(
            "Ultralytics reported exporting to {!r} but no such file exists.".format(produced)
        )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        destination = os.path.join(output_dir, os.path.basename(produced))
        if os.path.abspath(destination) != os.path.abspath(produced):
            shutil.move(produced, destination)
        produced = destination

    size_mb = os.path.getsize(produced) / (1024 * 1024)
    print("Wrote {} ({:.1f} MiB)".format(produced, size_mb))
    return produced


def verify_loadable_by_opencv(onnx_paths):
    """Best-effort check that OpenCV DNN can read what we just wrote.

    The export stage has `opencv-python` available as an Ultralytics dependency. It is a
    CPU-only wheel and cannot tell us anything about CUDA -- that is
    scripts/assert-cuda-build.py's job, in a different stage. What it CAN do is catch an
    ONNX graph the OpenCV importer rejects outright, here at build time rather than on a
    user's first inference request.
    """
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001 - this check is advisory
        print("Skipping OpenCV load check ({}: {})".format(type(exc).__name__, exc))
        return

    for path in onnx_paths:
        net = cv2.dnn.readNet(path)
        layers = len(net.getLayerNames())
        print("OpenCV DNN loaded {} ({} layers)".format(os.path.basename(path), layers))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="+", metavar="MODEL.pt",
                        help="Ultralytics .pt weights to export")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to place the .onnx files in "
                             "(default: alongside each .pt)")
    args = parser.parse_args(argv)

    written = []
    for pt_path in args.models:
        if not os.path.isfile(pt_path):
            raise SystemExit("No such file: {}".format(pt_path))
        written.append(export_one(pt_path, args.output_dir))

    verify_loadable_by_opencv(written)

    print("")
    print("Exported {} model(s):".format(len(written)))
    for path in written:
        print("  {}".format(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
