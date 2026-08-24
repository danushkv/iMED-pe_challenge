# Running the methods

This document gives commands, not precomputed artifacts. Set these paths once:

```bash
export IMEDPE_DATA_ROOT=/path/to/imed_pe
export IMEDPE_OUTPUT_ROOT=/path/to/generated_outputs
export XFEAT_ROOT="$PWD/third_party/accelerated_features"
export XFEAT_WEIGHTS="$XFEAT_ROOT/weights/xfeat.pt"
export LOFTR_ROOT="$PWD/third_party/EfficientLoFTR"
export LOFTR_CHECKPOINT="$LOFTR_ROOT/weights/eloftr_outdoor.ckpt"
```

Use `--split test` only after fixing all settings from TRAIN/session-balanced
validation. Examples below show TRAIN.

## Calibration

E1 and E2 calibrators automatically discover sessions in the supplied split:

```bash
uv run python scripts/run_method2_calibration.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --splits train \
  --max-keypoints 2048 \
  --num-samples 30 \
  --output-dir "$IMEDPE_OUTPUT_ROOT/calibration/e1"

uv run python scripts/run_method2b_calibration.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --splits train \
  --max-keypoints 2048 \
  --num-samples 30 \
  --output-dir "$IMEDPE_OUTPUT_ROOT/calibration/e2"
```

Calibration is per physical session, not per sequence. For an unseen split,
estimate it from that split's images at inference time; do not reuse a
different physical session's calibration.

## Method 1

```bash
uv run python scripts/run_stereo_vo.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --max-keypoints 2048 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method1"
```

For a sanity run, append `--sequence <name> --max-frames 50 --debug-frames 3`.

## Method 1.5A

Generate cross-camera anchors once, derive TRAIN confidence thresholds, then
run the frozen rotation correction:

```bash
uv run python scripts/generate_cross_camera_anchors.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --max-keypoints 2048 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method1_5/anchors"

uv run python scripts/derive_anchor_thresholds.py \
  --anchor-root "$IMEDPE_OUTPUT_ROOT/method1_5/anchors/train" \
  --lower-quantile 0.05 \
  --upper-quantile 0.95 \
  --output "$IMEDPE_OUTPUT_ROOT/method1_5/thresholds.json"

uv run python scripts/run_anchored_vo.py \
  --split train \
  --method1-root "$IMEDPE_OUTPUT_ROOT/method1" \
  --anchor-root "$IMEDPE_OUTPUT_ROOT/method1_5/anchors" \
  --thresholds "$IMEDPE_OUTPUT_ROOT/method1_5/thresholds.json" \
  --anchor-interval 5 \
  --alpha 0.75 \
  --max-correction-deg 5 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method1_5"
```

Thresholds are TRAIN-derived. Generate anchors for an evaluation split using
the same command with `--split test`, but reuse the frozen TRAIN thresholds.

## Method 2A and 2B

```bash
uv run python scripts/run_method2.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --max-keypoints 2048 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method2a"

uv run python scripts/run_method2b.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --max-keypoints 2048 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method2b"
```

Method 2A predictions are under `method2a/pure/<split>/...`; Method 2B uses
`method2b/<split>/...`.

## Method 3A/3B (VGGT)

Use the separate environment:

```bash
cd environments/vggt
uv sync
cd ../..

uv run --project environments/vggt python scripts/run_method3_vggt.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --variant 3b \
  --device cuda \
  --dtype auto \
  --preprocess-mode crop \
  --vggt-root third_party/vggt \
  --output-root "$IMEDPE_OUTPUT_ROOT/method3b"
```

Method 3A changes only `--variant 3a`. The official VGGT API retrieves
`facebook/VGGT-1B`; cache it before network-free execution.

## Method 4A

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
  --absolute-weight-scale 1 \
  --vo-weight-scale 1 \
  --huber-delta-steps 2.5 \
  --alignment-iterations 8 \
  --alignment-huber-mad 2.5 \
  --optimizer-max-nfev 200 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method4a1"
```

## XFeat

Verify the official sparse API, then create XFeat session calibration or run
the X1/X2/X3 ablations:

```bash
uv run python experiments/xfeat_matching/verify_xfeat.py \
  --xfeat-root "$XFEAT_ROOT" \
  --weights "$XFEAT_WEIGHTS" \
  --image0 /path/to/first.png \
  --image1 /path/to/second.png \
  --device cuda \
  --max-features 2048 \
  --output "$IMEDPE_OUTPUT_ROOT/xfeat/verification.json"

uv run python experiments/xfeat_matching/calibrate_xfeat.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --camera e1 \
  --splits train \
  --xfeat-root "$XFEAT_ROOT" \
  --weights "$XFEAT_WEIGHTS" \
  --max-features 2048 \
  --output-dir "$IMEDPE_OUTPUT_ROOT/xfeat/calibration/e1"

uv run python experiments/xfeat_matching/run_method2_xfeat.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --direction 2a \
  --variant x1 \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/xfeat/calibration/e1" \
  --xfeat-root "$XFEAT_ROOT" \
  --weights "$XFEAT_WEIGHTS" \
  --output-root "$IMEDPE_OUTPUT_ROOT/xfeat/method2a_x1"
```

X1 changes session calibration only; X2 changes per-frame inference only; X3
changes both. The selected hybrid used XFeat E1 calibration and otherwise kept
the ALIKED/LightGlue Method 4A stack.

## EfficientLoFTR multi-view

The selected experiment uses official full/fp32 EfficientLoFTR, confidence
0.8, and 3 px mutual-nearest association in original image coordinates:

```bash
uv run python experiments/efficient_loftr_multiview/run_method2a.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --loftr-root "$LOFTR_ROOT" \
  --checkpoint "$LOFTR_CHECKPOINT" \
  --device cuda \
  --model-type full \
  --precision fp32 \
  --confidence-threshold 0.8 \
  --epsilon 3 \
  --output-root "$IMEDPE_OUTPUT_ROOT/loftr_mv/method2a"

uv run python experiments/efficient_loftr_multiview/run_method2b.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --loftr-root "$LOFTR_ROOT" \
  --checkpoint "$LOFTR_CHECKPOINT" \
  --device cuda \
  --model-type full \
  --precision fp32 \
  --confidence-threshold 0.8 \
  --epsilon 3 \
  --output-root "$IMEDPE_OUTPUT_ROOT/loftr_mv/method2b"
```

For submission-safe experiments, `--frame-zero-fallback-root` may point to the
corresponding original Method 2 prediction root. Such outputs must be labeled
as fallback variants rather than pure LoFTR-MV.

## Evaluation

```bash
uv run python scripts/evaluate_ate.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --pred-root "$IMEDPE_OUTPUT_ROOT/method4a1" \
  --json-out "$IMEDPE_OUTPUT_ROOT/evaluation/method4a1_train.json"

uv run python experiments/session_holdout_validation/evaluate_loso.py \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --prediction baseline="$IMEDPE_OUTPUT_ROOT/baseline" \
  --prediction method1="$IMEDPE_OUTPUT_ROOT/method1" \
  --prediction method2a="$IMEDPE_OUTPUT_ROOT/method2a/pure" \
  --prediction method2b="$IMEDPE_OUTPUT_ROOT/method2b" \
  --prediction method4a="$IMEDPE_OUTPUT_ROOT/method4a1" \
  --baseline-name baseline \
  --output-root "$IMEDPE_OUTPUT_ROOT/evaluation/loso_train"
```

