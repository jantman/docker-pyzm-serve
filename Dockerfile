# syntax=docker/dockerfile:1.7
#
# GPU-accelerated pyzm.serve inference gateway.
#
# Three stages, in the order that keeps the expensive one cached (FR-019):
#
#   opencv-build   OpenCV + opencv_contrib compiled from source with CUDA and cuDNN.
#                  Multi-hour, and the only stage that matters for build time. Nothing
#                  below it may invalidate it -- which is why the model export happens in a
#                  *separate* stage rather than after it, and why .dockerignore keeps the
#                  documentation tree out of the build context entirely.
#
#   model-export   Throwaway. Ultralytics + CPU-only PyTorch on a slim Python base, used to
#                  turn the pinned .pt weights into .onnx and to fetch the Darknet weights.
#                  It exists to CONTAIN Ultralytics: Ultralytics depends on opencv-python,
#                  the PyPI wheel that shadows a source-built CUDA OpenCV. Its
#                  site-packages never reach the shipped image (research R4).
#
#   runtime        What ships. CUDA *runtime* base -- no nvcc, no compiler toolchain
#                  (FR-018). Receives only the OpenCV install tree and the model files.
#
# Every external input below is pinned to an immutable identifier (Constitution Principle
# II). Base images by digest, git checkouts by full commit SHA, model weights by SHA-256
# checksum verified during the build. Nothing is fetched at run time (Principle VI).

# =============================================================================
# Pinned inputs -- see specs/001-pyzm-serve-gpu-image/data-model.md
# =============================================================================

# --- Base images (by digest, never by tag) -----------------------------------
# nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ARG CUDA_DEVEL_IMAGE=nvidia/cuda@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42
# nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ARG CUDA_RUNTIME_IMAGE=nvidia/cuda@sha256:2fcc4280646484290cc50dce5e65f388dd04352b07cbe89a635703bd1f9aedb6
# python:3.12-slim-bookworm -- the export stage only. Deliberately NOT the CUDA base: this
# stage needs no GPU, and putting it on the CUDA image would drag several GB of wheels in.
ARG EXPORT_BASE_IMAGE=python@sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134

# --- Git checkouts (full commit SHAs -- tags are mutable) --------------------
# opencv/opencv tag 4.12.0
ARG OPENCV_REF=cbee6841638edb6fbc8110df7cd52bb8e3d66211
# opencv/opencv_contrib tag 4.12.0
ARG OPENCV_CONTRIB_REF=7deb35fde4d38d73b6173c7ab2aeddc8df5a89e3
# pyzmNg. Normally the tag object of a ZoneMinder/pyzmNg release; both the repository and
# the ref are ARGs so a fork can be tested without editing the pip line below.
#
# ###########################################################################
# # TEMPORARY -- NOT RELEASABLE. This points at a FORK, not upstream:       #
# #   jantman/pyzmNg @ integration/66-68, a merge of the two open PRs:      #
# #     ZoneMinder/pyzmNg#67 (issues/66) -- GPU-fallback retry, the         #
# #       `processor` key on /models, and --no-cpu-fallback, all of which   #
# #       this image depends on (see issue #1).                             #
# #     ZoneMinder/pyzmNg#69 (issues/68) -- zone_match_strategy. NOT used   #
# #       by this image: zone filtering is client-side (pyzm.ml.filters),   #
# #       and no DetectorConfig crosses /infer. Pinned here only so the     #
# #       gateway and docker-zoneminder run one identical pyzm build.       #
# #   The two PR branches are independent off master; the merge exists      #
# #   solely to give an image a single SHA and is never itself PR'd.        #
# #                                                                         #
# # Both lines MUST go back to ZoneMinder/pyzmNg at a release tag before    #
# # any tag is cut here. Principle IV: a release promises a reproducible    #
# # artifact, and a fork branch is not one -- the SHA below is immutable    #
# # but the fork itself can vanish or be force-pushed past.                 #
# ###########################################################################
ARG PYZM_REPO=https://github.com/jantman/pyzmNg.git
ARG PYZM_REF=271bf98c33c28edca231c0f617d79887acd3a001

# --- Model inputs ------------------------------------------------------------
# The ultralytics/assets release the two .pt files come from.
ARG YOLO11_ASSETS_REF=v8.4.0
# AlexeyAB/darknet release asset. A release tag, not a branch.
ARG YOLOV4_WEIGHTS_URL=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights

# --- Export-stage Python packages (exact ==, export stage only) --------------
ARG ULTRALYTICS_VERSION=8.4.120
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG ONNX_VERSION=1.22.0
ARG ONNXSLIM_VERSION=0.1.95
ARG ONNXRUNTIME_VERSION=1.28.0

# --- NumPy: pinned, and pinned identically in BOTH stages that matter ---------
# This is not a stylistic pin. NumPy 2.0 broke the C ABI: an extension compiled against
# NumPy 1.x refuses to import under NumPy 2.x ("A module that was compiled using NumPy 1.x
# cannot be run in NumPy 2.x"). Ubuntu 22.04's python3-numpy is 1.21, while pyzm[serve]
# depends on an unpinned `numpy` and pip would resolve that to 2.x -- so building cv2
# against the apt NumPy and then installing pyzm produces an image whose `import cv2`
# fails outright. Pinning one version and using it in the compile stage AND the runtime
# stage is what keeps the bindings loadable.
#
# 2.2.6 is the newest NumPy that still publishes cp310 wheels, and Ubuntu 22.04's Python is
# 3.10. Moving off 22.04 is what unpins this.
ARG NUMPY_VERSION=2.2.6

# --- CUDA build configuration ------------------------------------------------
# 6.1 Pascal, 7.5 Turing, 8.6 consumer Ampere, 8.9 Ada. Four generations rather than one,
# because an image that only works on the author's GPU is not usable by anyone else
# (Constitution Principle V, FR-017).
ARG CUDA_ARCH_BIN="6.1;7.5;8.6;8.9"
# NOT optional decoration. CUDA_ARCH_BIN emits SASS for exactly the four architectures
# above and NOTHING for anything newer -- a Blackwell card would fail outright with "no
# kernel image is available for execution on the device". CUDA_ARCH_PTX emits the
# intermediate representation such a card JIT-compiles at first load instead, which is what
# FR-017's forward-compatibility clause requires and what contracts/container-interface.md
# section 9 promises. The one-off JIT delay is why the HEALTHCHECK start period below is
# generous (research R3, R10).
ARG CUDA_ARCH_PTX="8.9"
# The gate's own test subject. Building with OPENCV_CUDA=OFF must produce a build that
# FAILS at scripts/assert-cuda-build.py -- quickstart Scenario 1, T024. A gate nobody has
# watched fail is not known to work, and that is precisely what cost the predecessor stack
# a year of CPU inference.
ARG OPENCV_CUDA=ON


# =============================================================================
# Stage 1: opencv-build -- OpenCV with CUDA, from source
# =============================================================================
FROM ${CUDA_DEVEL_IMAGE} AS opencv-build

ARG OPENCV_REF
ARG OPENCV_CONTRIB_REF
ARG CUDA_ARCH_BIN
ARG CUDA_ARCH_PTX
ARG OPENCV_CUDA
ARG NUMPY_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build dependencies. Ubuntu 22.04 ships Python 3.10, which satisfies pyzm's
# `python_requires>=3.10` -- and, unlike Debian 12, predates PEP 668's externally-managed
# marker, so pip works without a venv or --break-system-packages.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ccache \
        pkg-config \
        python3-dev \
        python3-pip \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libwebp-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# NumPy from pip at the pinned version, NOT apt's python3-numpy. The OpenCV Python bindings
# are compiled against whatever NumPy headers are present here, and that choice has to match
# the NumPy the runtime stage ends up with or `import cv2` fails on an ABI mismatch. See the
# NUMPY_VERSION comment at the top of this file.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install "numpy==${NUMPY_VERSION}"

WORKDIR /src

# Clone at the pinned SHAs. `git init` + `fetch <sha>` rather than `clone --branch` because
# a SHA is not a ref and cannot be cloned directly; this also avoids fetching history we
# will never use.
RUN git init opencv \
    && git -C opencv remote add origin https://github.com/opencv/opencv.git \
    && git -C opencv fetch --depth 1 origin "${OPENCV_REF}" \
    && git -C opencv checkout FETCH_HEAD \
    && git init opencv_contrib \
    && git -C opencv_contrib remote add origin https://github.com/opencv/opencv_contrib.git \
    && git -C opencv_contrib fetch --depth 1 origin "${OPENCV_CONTRIB_REF}" \
    && git -C opencv_contrib checkout FETCH_HEAD

# Configure.
#
# opencv_contrib is REQUIRED, not a nice-to-have, for two independent reasons:
#   - OPENCV_DNN_CUDA=ON depends on the `cudev` module, which lives in contrib. Omit it and
#     you silently get a CUDA-less DNN -- exactly the failure this project exists to prevent.
#   - `cudaarithm` is what puts the `cv2.cuda` namespace into the Python bindings, and
#     therefore `cv2.cuda.getCudaEnabledDeviceCount()`. Both the entrypoint GPU preflight
#     and quickstart Scenario 5b call it, which makes contrib load-bearing for verification
#     rather than incidental.
#
# BUILD_LIST is the primary lever on build time (research R2, R8): ten modules rather than
# sixty, multiplied by four GPU architectures, is the difference between fitting inside a
# GitHub runner's 6-hour job cap and not. Each entry earns its place -- `dnn` is the engine,
# `imgcodecs` decodes the uploaded frame, `imgproc` does the letterbox resize, `cudev` is
# DNN-CUDA's dependency, `cudaarithm` provides cv2.cuda, and `python3` is how any of it is
# reachable.
#
# The CUDA-dependent flags all follow ${OPENCV_CUDA} rather than being hardcoded ON. That is
# deliberate: with OPENCV_CUDA=OFF the configure step must SUCCEED and produce a clean
# CPU-only build, so that the failure lands on the assertion below -- which is the thing
# under test in quickstart Scenario 1. Hardcoding OPENCV_DNN_CUDA=ON would fail at cmake
# instead and prove nothing about the gate.
RUN cmake -S opencv -B build \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/src/opencv_contrib/modules \
        -D BUILD_LIST=core,imgproc,imgcodecs,videoio,dnn,python3,cudev,cudaarithm,cudawarping,cudaimgproc \
        -D WITH_CUDA=${OPENCV_CUDA} \
        -D WITH_CUDNN=${OPENCV_CUDA} \
        -D OPENCV_DNN_CUDA=${OPENCV_CUDA} \
        -D WITH_CUBLAS=${OPENCV_CUDA} \
        -D CUDA_ARCH_BIN=${CUDA_ARCH_BIN} \
        -D CUDA_ARCH_PTX=${CUDA_ARCH_PTX} \
        -D BUILD_opencv_python3=ON \
        -D OPENCV_PYTHON3_INSTALL_PATH=/usr/local/lib/python3.10/dist-packages \
        -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_JAVA=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D CMAKE_C_COMPILER_LAUNCHER=ccache \
        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -D CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
    && cmake -B build -L -N | grep -iE "CUDA|cudnn" || true

# Compile and install into a staging prefix the runtime stage can COPY --from.
#
# DESTDIR staging rather than a different CMAKE_INSTALL_PREFIX: the libraries keep the
# /usr/local paths they were linked against, so the runtime stage gets working rpaths and
# a plain `ldconfig` is enough.
#
# ccache is persisted through a BuildKit cache mount, which survives across builds locally
# and through the registry-backed cache in CI (research R8).
RUN --mount=type=cache,target=/root/.cache/ccache,sharing=locked \
    export CCACHE_DIR=/root/.cache/ccache \
    && cmake --build build --parallel "$(nproc)" \
    && cmake --install build --prefix /usr/local \
    && DESTDIR=/staging cmake --install build \
    && ldconfig \
    && ccache --show-stats || true

# THE GATE (FR-008).
#
# Runs against the OpenCV that was just built, and fails the build if it reports no CUDA
# support. This is the check Constitution Principle I is built on, and it works on a
# GPU-less runner because it tests what was COMPILED, not what is AVAILABLE (FR-027).
#
# Bind-mounted rather than COPYed: the same file is executed again in the runtime stage
# below and a third time by scripts/verify-published.sh against the image pulled back from
# the registry, so the three checks cannot drift apart -- but it is not baked into the
# shipped image, which carries no verification tooling of its own.
RUN --mount=type=bind,source=scripts/assert-cuda-build.py,target=/tmp/assert-cuda-build.py \
    python3 /tmp/assert-cuda-build.py


# =============================================================================
# Stage 2: model-export -- .pt -> .onnx, plus the Darknet weights
# =============================================================================
# Throwaway. Nothing installed here reaches the shipped image; only the resulting model
# files are copied out. This is where the Ultralytics/opencv-python collision is contained.
FROM ${EXPORT_BASE_IMAGE} AS model-export

ARG YOLO11_ASSETS_REF
ARG YOLOV4_WEIGHTS_URL
ARG ULTRALYTICS_VERSION
ARG TORCH_VERSION
ARG TORCHVISION_VERSION
ARG ONNX_VERSION
ARG ONNXSLIM_VERSION
ARG ONNXRUNTIME_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp/ultralytics

# The X11/GL shared libraries are here because Ultralytics imports cv2 unconditionally at
# module load, and the PyPI `opencv-python` wheel it depends on links against them even
# when nothing ever opens a window. python:3.12-slim carries none of them, so the export
# fails with `ImportError: libxcb.so.1`.
#
# This is a headless build stage, so those libraries are pure dead weight -- and that is
# fine, because NOTHING FROM THIS STAGE SHIPS. Only the exported .onnx and the downloaded
# .weights are copied out. Adding them here is the boring fix; the clever alternative
# (swapping in opencv-python-headless behind Ultralytics' back) would leave two packages
# both claiming to provide cv2 in the stage whose entire job is to contain that mess.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
        libgl1 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-ONLY PyTorch. The default index would pull several gigabytes of CUDA wheels for an
# export that runs entirely on the CPU, inflating this stage and its cache for nothing.
#
# onnx / onnxslim / onnxruntime are pinned explicitly rather than left to Ultralytics'
# on-demand installer: YOLO_AUTOINSTALL=false above disables that installer precisely
# because it would fetch unpinned versions at build time, which Principle II forbids.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
    && pip install \
        "ultralytics==${ULTRALYTICS_VERSION}" \
        "onnx==${ONNX_VERSION}" \
        "onnxslim==${ONNXSLIM_VERSION}" \
        "onnxruntime==${ONNXRUNTIME_VERSION}"

WORKDIR /models

# Fetch the pinned .pt weights and VERIFY THEM BEFORE USE (FR-014, PI-2).
#
# --ignore-missing is required, not lazy: models/checksums.sha256 also records
# yolov4.weights, which is fetched in the next step and is not on disk yet. sha256sum still
# fails the build if any file that IS present mismatches, and errors out if nothing at all
# was verified, so the gate cannot silently pass.
#
# A model that changes silently produces detection differences indistinguishable from a
# code regression. This check is the only thing standing between a user and that.
RUN --mount=type=bind,source=models/checksums.sha256,target=/tmp/checksums.sha256 \
    curl -fsSL -o yolo11m.pt \
        "https://github.com/ultralytics/assets/releases/download/${YOLO11_ASSETS_REF}/yolo11m.pt" \
    && curl -fsSL -o yolo11s.pt \
        "https://github.com/ultralytics/assets/releases/download/${YOLO11_ASSETS_REF}/yolo11s.pt" \
    && sha256sum -c --ignore-missing /tmp/checksums.sha256

# Export to ONNX at imgsz=640 with nms=False -- both matched to what upstream's ONNX
# backend assumes. See scripts/export-onnx.py for why each parameter is what it is.
RUN --mount=type=bind,source=scripts/export-onnx.py,target=/tmp/export-onnx.py \
    python3 /tmp/export-onnx.py yolo11m.pt yolo11s.pt --output-dir /out/ultralytics

# YOLOv4 Darknet weights, from a release asset (not a branch), checksum-verified the same
# way. Retained as the fallback known to work with OpenCV DNN's CUDA backend; the
# Constitution names it explicitly under Models.
RUN --mount=type=bind,source=models/checksums.sha256,target=/tmp/checksums.sha256 \
    mkdir -p /out/yolov4 \
    && curl -fsSL -o /out/yolov4/yolov4.weights "${YOLOV4_WEIGHTS_URL}" \
    && cd /out/yolov4 \
    && sha256sum -c --ignore-missing /tmp/checksums.sha256

# The two small text files are vendored in this repository rather than fetched, because
# upstream publishes them only from a branch ref (see models/PROVENANCE.md).
COPY models/yolov4.cfg models/coco.names /out/yolov4/


# =============================================================================
# Stage 3: runtime -- what actually ships
# =============================================================================
FROM ${CUDA_RUNTIME_IMAGE} AS runtime

ARG PYZM_REPO
ARG PYZM_REF
ARG NUMPY_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Minimal Python runtime. No build-essential, no cmake, no nvcc -- the CUDA *runtime* base
# carries the CUDA and cuDNN shared libraries without the toolkit, which is exactly what
# FR-018 asks for. `git` is here only because pip needs it to install pyzm from a pinned
# commit, and is removed again below.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libjpeg8 \
        libpng16-16 \
        libtiff5 \
        libwebp7 \
        libwebpmux3 \
        libwebpdemux2 \
        libopenjp2-7 \
        libgomp1 \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

# The SAME NumPy the bindings were compiled against, installed before anything else can
# resolve a different one. NumPy 2.0 broke the C ABI, so a mismatch here is not a
# performance problem -- `import cv2` simply fails. See the NUMPY_VERSION comment above.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip3 install "numpy==${NUMPY_VERSION}"

# The OpenCV install tree, and nothing else, from the compile stage.
COPY --from=opencv-build /staging/usr/local /usr/local
RUN ldconfig

# The model tree (data-model.md, "Entity: Model Artifact"). Names are file stems, so
# `--models yolo11m` resolves to ultralytics/yolo11m.onnx by upstream's discovery rules.
COPY --from=model-export /out/ /var/lib/zmeventnotification/models/

# -----------------------------------------------------------------------------
# The opencv-python dist-info shim -- DO NOT DELETE THIS (FR-009)
# -----------------------------------------------------------------------------
# This writes a *fake* opencv_python-<version>.dist-info into site-packages so that pip
# believes opencv-python is already installed and will never fetch the PyPI wheel.
#
# WHAT IT PREVENTS: the PyPI `opencv-python` wheel is a CPU-only build. If anything ever
# pulls it in, it lands in site-packages ahead of the CUDA-enabled cv2 compiled above and
# SILENTLY SHADOWS IT. Nothing errors. `import cv2` still works, pyzm still reports
# processor:gpu, and every frame is then decoded and inferred on the CPU. That precise
# failure ran the predecessor stack -- jantman/docker-zm-mlapi -- on the CPU for over a year
# before anyone noticed, because the only symptom is that detection feels slow.
#
# WHY IT LOOKS UNNECESSARY: it is, today. pyzm[serve] declares neither Ultralytics nor
# opencv-python (research R4), so nothing currently wants the wheel. That is exactly why it
# is here: it makes the invariant hold even if a future transitive dependency, or a future
# maintainer reaching for the more obvious-looking [full] extra, introduces something that
# does. A tidy-up that removes this "redundant" block restores a year-long silent bug.
#
# The technique is upstream's own, from scripts/setup_venv.sh:125-155 at pyzmNg v2.5.1:
#   https://github.com/ZoneMinder/pyzmNg/blob/v2.5.1/scripts/setup_venv.sh
#
# scripts/assert-cuda-build.py at the end of this stage is the backstop, not the solution.
RUN CV2_VERSION="$(python3 -c 'import cv2; print(cv2.__version__)')" \
    && SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')" \
    && DIST_DIR="${SITE_PACKAGES}/opencv_python-${CV2_VERSION}.dist-info" \
    && mkdir -p "${DIST_DIR}" \
    && printf 'Metadata-Version: 2.1\nName: opencv-python\nVersion: %s\nSummary: Shim - real cv2 is provided by the CUDA source build in this image\n' \
        "${CV2_VERSION}" > "${DIST_DIR}/METADATA" \
    && : > "${DIST_DIR}/RECORD" \
    && echo "opencv-python" > "${DIST_DIR}/top_level.txt" \
    && echo "Wheel-Version: 1.0" > "${DIST_DIR}/WHEEL" \
    && echo "opencv-python shim created for cv2 ${CV2_VERSION} in ${SITE_PACKAGES}"

# -----------------------------------------------------------------------------
# pyzm -- [serve], NEVER [full]
# -----------------------------------------------------------------------------
# [full] looks like the obviously-safer choice and is the trap. Read from setup.py at
# v2.5.1: `ultralytics>=8.3` appears in [train] and [full] only. Ultralytics depends on
# `opencv-python`, so installing [full] here would fetch the CPU-only wheel and shadow the
# source build -- the exact failure the shim above exists to catch. [serve] declares
# neither (research R4).
#
# [serve] also declares no OpenCV dependency at all; pyzm.ml simply expects `import cv2` to
# work, which is what the source build provides. That is why cv2 is installed first.
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir \
        "pyzm[serve] @ git+${PYZM_REPO}@${PYZM_REF}" \
        "numpy==${NUMPY_VERSION}" \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

# THE GATE, AGAIN (FR-009).
#
# Deliberately re-run here, after the shim and after pip has resolved pyzm's entire
# dependency tree, against the OpenCV that the SHIPPED image will actually load. The
# assertion in the compile stage proves the build was configured correctly; this one proves
# nothing between there and here disturbed it. Same file, so the two cannot drift apart.
RUN --mount=type=bind,source=scripts/assert-cuda-build.py,target=/tmp/assert-cuda-build.py \
    python3 /tmp/assert-cuda-build.py

# -----------------------------------------------------------------------------
# Runtime metadata
# -----------------------------------------------------------------------------
COPY entrypoint.sh /opt/entrypoint.sh
COPY scripts/warmup.py /opt/warmup.py
RUN chmod 0555 /opt/entrypoint.sh /opt/warmup.py

# Defaults for the whole variable table in contracts/container-interface.md section 5.
#
# Two deliberately differ from upstream and are called out in the README, because a user
# following upstream's documentation would otherwise be surprised (RC-1):
#   PYZM_SERVE_PROCESSOR  upstream cpu -> gpu here. This is a GPU image; a user who
#                         silently gets CPU has the predecessor's bug back as a feature.
#   PYZM_SERVE_MODELS     upstream yolo11s -> yolo11m here, the primary model per the spec.
#
# PYZM_SERVE_DEBUG, _AUTH, _AUTH_PASSWORD, _TOKEN_SECRET and _ALLOW_CPU are deliberately
# NOT declared. They are unset by default and the entrypoint tests for emptiness; declaring
# them empty here would work but would advertise them as configured, which they are not.
ENV PYZM_SERVE_MODELS=yolo11m \
    PYZM_SERVE_PROCESSOR=gpu \
    PYZM_SERVE_PORT=5000 \
    PYZM_SERVE_HOST=0.0.0.0 \
    PYZM_SERVE_BASE_PATH=/var/lib/zmeventnotification/models \
    PYZM_SERVE_WORKERS=1 \
    PYZM_SERVE_AUTH_USER=admin

EXPOSE 5000

# Non-root. Nothing in this image needs to write to its own filesystem: the models are
# read-only, there is no state, no database and no cache to warm.
RUN groupadd --system --gid 10001 pyzm \
    && useradd --system --uid 10001 --gid 10001 --home-dir /nonexistent \
        --no-create-home --shell /usr/sbin/nologin pyzm
USER 10001:10001

# No VOLUME is declared, on purpose. The image needs no writable storage (FR-033), and a
# VOLUME here would create an anonymous volume on every `docker run` for nothing.

# Healthcheck against the CONFIGURED port, not a hardcoded 5000 -- otherwise every operator
# who moves the port gets a container that is permanently unhealthy while serving perfectly
# (research R10). ${PYZM_SERVE_PORT} is expanded by the shell at check time, so it follows
# `docker run -e PYZM_SERVE_PORT=...`.
#
# urllib rather than curl: there is no curl in the runtime base image, and adding one for a
# healthcheck is a wasteful dependency when Python is guaranteed present.
#
# The 180s start period is sized for the worst SUPPORTED case, not the typical one: a GPU
# newer than any architecture compiled above JIT-compiles the PTX on first model load, which
# takes tens of seconds. contracts/container-interface.md section 9 promises those cards
# work; too short a start period would turn that promise into a crash loop.
#
# /health ALONE IS NOT ENOUGH (issue #1). It answers {"status":"ok","models_loaded":true}
# whether inference is running on the GPU or has degraded to the CPU, so a container that
# had silently become five to eight times slower stayed green for its entire life. The
# second call compares each model's live `processor` against its `requested_processor` --
# upstream #67's two new keys -- and fails the check when they differ. That is the only
# thing that makes "this gateway has degraded to CPU" alertable rather than a latency graph
# somebody eventually notices.
#
# Both keys are read with .get(): against a pyzm that predates #67 they are absent, every
# model is skipped, and the check degrades to exactly the /health test it replaced rather
# than reporting a false failure.
#
# `/models` needs no token even when PYZM_SERVE_AUTH is on (only `/infer` is behind auth
# upstream), so this works in every auth configuration.
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD python3 -c "import json,os,sys,urllib.request as u; \
b='http://127.0.0.1:'+os.environ.get('PYZM_SERVE_PORT','5000'); \
sys.exit(1) if u.urlopen(b+'/health', timeout=8).status!=200 else None; \
d=[m for m in json.load(u.urlopen(b+'/models', timeout=8))['models'] \
if m.get('processor') and m.get('requested_processor') \
and m['processor']!=m['requested_processor']]; \
print('DEGRADED: '+'; '.join(m['name']+' is running on '+m['processor']+', not the requested '+m['requested_processor'] for m in d)) if d else None; \
sys.exit(1 if d else 0)"

ENTRYPOINT ["/opt/entrypoint.sh"]
