# iMED-PE Challenge Methods

Research code for multi-view camera pose estimation developed for the iMED
Pose Estimation Challenge. The repository contains classical stereo geometry,
visual odometry, robust trajectory fusion, and zero-shot learned geometry
experiments.

> **Challenge embargo:** keep this repository private until the challenge
> organizers permit publication of methods and results. The dataset, generated
> predictions, calibration caches, plots containing challenge images, and model
> weights are intentionally not included.

## Implemented methods

| Name | Summary |
|---|---|
| Baseline | ALIKED + LightGlue cross-endoscope essential matrix |
| Method 1 | Endoscope-2 stereo visual odometry with temporal PnP |
| Method 1.5A | Method 1 with independent cross-endoscope rotation anchors |
| Method 2A | Per-frame E1 stereo reconstruction to E2-L PnP |
| Method 2B | Reverse per-frame E2 stereo reconstruction to E1-L PnP |
| Method 3A/3B | Zero-shot VGGT four-view and reference-anchored eight-view poses |
| Method 4A | Robust fusion of Method 2A/2B absolute poses and Method 1 VO edges |
| XFeat | Sparse-XFeat front-end ablations and XFeat-calibrated Method 4A |
| EfficientLoFTR | Pairwise and pure three-view LoFTR experiments for Method 2/4A |
| RoMa-2A | Exact-index ALIKED source pixels warped into E1-R and E2-L by full RoMa |
| Method 5-R | Geometry-only E1 stereo correction followed by RoMa-2A and frozen Method 4A |

The exact transform conventions are documented in
[`docs/TRANSFORMS.md`](docs/TRANSFORMS.md). Frozen experiment settings are in
[`configs/`](configs/).

## Repository layout

```text
configs/        frozen method settings
docs/           dataset contract, methods, results, and reproducibility notes
environments/   separate VGGT environment
experiments/    XFeat, EfficientLoFTR, RoMa, Method 5-R, and LOSO code
scripts/        calibration, inference, evaluation, and visualization CLIs
src/imcpe/      shared geometry and method implementations
tests/          dependency-free/synthetic safety checks
third_party/    revision manifest and setup helper (no vendored weights)
```

## Installation

The local research runs used the `pe-basline` Python 3.10 uv environment,
captured by the root lockfile. Challenge Docker runs used the smaller CUDA 11.8
stack recorded separately in `environments/docker/`. Install
[uv](https://docs.astral.sh/uv/) and run:

```bash
git clone https://github.com/<account>/iMED-pe_challenge.git
cd iMED-pe_challenge
uv sync --extra dev
uv run python scripts/check_install.py
```

For exact reproduction after cloning, prefer `uv sync --extra dev --frozen`.

The committed `uv.lock` fixes the Python dependency graph. Model source and
weights are handled separately because they have their own licenses:

```bash
bash third_party/setup_models.sh --lightglue
bash third_party/setup_models.sh --xfeat
bash third_party/setup_models.sh --loftr
bash third_party/setup_models.sh --roma
```

The helper clones exact commits and explains where an official checkpoint must
be placed. It does not silently download license-gated files. See
[`THIRD_PARTY.md`](THIRD_PARTY.md).

Cache the official ALIKED/LightGlue weights explicitly before offline use:

```bash
uv run python scripts/cache_lightglue_weights.py --max-keypoints 2048
```

VGGT is deliberately isolated because its current dependency requirements are
newer than the evaluated classical stack:

```bash
cd environments/vggt
uv sync
cd ../..
bash third_party/setup_models.sh --vggt
```

## Dataset contract

Pass the dataset at runtime; never edit source paths. The expected layout is:

```text
<DATA_ROOT>/
  train/<sequence>/
  test/<sequence>/
```

Each sequence contains `K.txt`, optional evaluation-only `pose.txt`, and:

```text
endoscope1/L/frame_XXXXXX.png
endoscope1/R/frame_XXXXXX.png
endoscope2/L/frame_XXXXXX.png
endoscope2/R/frame_XXXXXX.png
```

Validate a local dataset without running a model:

```bash
uv run python scripts/validate_dataset.py --data-root /path/to/imed_pe
```

More detail is in [`docs/DATASET.md`](docs/DATASET.md).
Full provenance and fresh-clone checks are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Reproduce the main pipeline

All examples use shell variables so no machine-specific path enters the code:

```bash
export IMEDPE_DATA_ROOT=/path/to/imed_pe
export IMEDPE_OUTPUT_ROOT=/path/to/generated_outputs
```

### 1. Session calibration and Method 1

Estimate one fixed E2 stereo transform per session, then run stereo VO:

```bash
uv run python scripts/run_stereo_calibration.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --session 001 \
  --splits train \
  --output-dir "$IMEDPE_OUTPUT_ROOT/calibration/e2"

uv run python scripts/run_stereo_vo.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --max-keypoints 2048 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method1"
```

Repeat calibration for every discovered session. Method 1.5 anchor generation
and fusion are described in [`docs/RUN_METHODS.md`](docs/RUN_METHODS.md).

### 2. Method 2A and Method 2B

```bash
uv run python scripts/run_method2_calibration.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --splits train \
  --output-dir "$IMEDPE_OUTPUT_ROOT/calibration/e1"

uv run python scripts/run_method2.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --output-root "$IMEDPE_OUTPUT_ROOT/method2a"

uv run python scripts/run_method2b.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --output-root "$IMEDPE_OUTPUT_ROOT/method2b"
```

### 3. Method 4A fusion

Method 4A consumes saved predictions; it does not rerun feature extraction:

```bash
uv run python scripts/run_method4a.py \
  --split train \
  --method1-root "$IMEDPE_OUTPUT_ROOT/method1" \
  --method2a-root "$IMEDPE_OUTPUT_ROOT/method2a/pure" \
  --method2b-root "$IMEDPE_OUTPUT_ROOT/method2b" \
  --rotation-root "$IMEDPE_OUTPUT_ROOT/method1_5" \
  --mode 4a1 \
  --confidence-version v1 \
  --lambda-vo 5 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method4a1"
```

The XFeat, EfficientLoFTR, and VGGT commands are in
[`docs/RUN_METHODS.md`](docs/RUN_METHODS.md).

RoMa-2A and Method 5-R use the separate locked environment under
`environments/roma/`. Their complete frozen commands are also documented in
[`docs/RUN_METHODS.md`](docs/RUN_METHODS.md).

## Evaluation and visualizations

Evaluation uses the repository's challenge-compatible Horn/Sim(3) alignment,
ATE, and RPE implementation:

```bash
uv run python scripts/evaluate_ate.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --pred-root "$IMEDPE_OUTPUT_ROOT/method4a1"
```

Create a publication-style grid with synchronized endoscope frames and aligned
trajectories:

```bash
uv run python scripts/make_comparison_video.py \
  --sequence-dir "$IMEDPE_DATA_ROOT/train/<sequence>" \
  --prediction method1="$IMEDPE_OUTPUT_ROOT/method1/train/<sequence>/pose.txt" \
  --prediction method2b="$IMEDPE_OUTPUT_ROOT/method2b/train/<sequence>/pose.txt" \
  --prediction method4a="$IMEDPE_OUTPUT_ROOT/method4a1/train/<sequence>/pose.txt" \
  --output "$IMEDPE_OUTPUT_ROOT/figures/<sequence>.mp4"
```

These media files remain local and are ignored by Git. Result provenance and
the current comparison tables are in [`docs/RESULTS.md`](docs/RESULTS.md).

## Checkpoints

There are no challenge-trained checkpoints. All challenge-specific calibration
and anchor caches are estimated independently per session at inference time.
ALIKED/LightGlue, XFeat, EfficientLoFTR, RoMa, and VGGT use official third-party
pretrained weights. Do not re-upload those weights under this project's name;
link to their official model cards/releases and preserve their licenses.

## Publication checklist

Before changing this repository from private to public:

1. Confirm the challenge embargo and dataset visualization rules.
2. Replace the provisional `LICENSE` notice after clarifying the baseline license.
3. Replace the GitHub account placeholder in `CITATION.cff`.
4. Do not commit dataset frames, pose files, predictions, calibration caches, or weights.
5. Re-run the reproducibility commands from a fresh clone and record hardware.

The standalone Git initialization and push commands are in
[`docs/PUBLISHING.md`](docs/PUBLISHING.md).
