# Running the methods

This document gives commands, not precomputed artifacts. Set these paths once:

```bash
export IMEDPE_DATA_ROOT=/path/to/imed_pe
export IMEDPE_OUTPUT_ROOT=/path/to/generated_outputs
export XFEAT_ROOT="$PWD/third_party/accelerated_features"
export XFEAT_WEIGHTS="$XFEAT_ROOT/weights/xfeat.pt"
export LOFTR_ROOT="$PWD/third_party/EfficientLoFTR"
export LOFTR_CHECKPOINT="$LOFTR_ROOT/weights/eloftr_outdoor.ckpt"
export ROMA_ROOT="$PWD/third_party/RoMa"
export ROMA_TORCH_HOME="$ROMA_ROOT/.torch"
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

## RoMa-2A

Install the separate frozen environment and verify the official checkout and
weights before inference:

```bash
uv sync --project environments/roma --frozen
bash third_party/setup_models.sh --roma
PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python \
  scripts/check_install.py --require-roma
```

Generate exact-index RoMa-2A observations with the TRAIN-frozen certainty and
cycle thresholds:

```bash
PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.roma_2a.run_method2a_roma \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --roma-root "$ROMA_ROOT" \
  --torch-home "$ROMA_TORCH_HOME" \
  --device cuda \
  --max-source-points 2048 \
  --certainty-threshold 0.20 \
  --cycle-threshold-px 2 \
  --debug-frames 0 \
  --output-root "$IMEDPE_OUTPUT_ROOT/roma2a"
```

## Method 5-R and frozen RoMa/4A fusion

Method 5-R estimates one small E1 stereo correction per physical session. The
same global validity rule is used for every session, and an invalid fit falls
back to the original calibration. These commands do not read ground truth.

Cache deterministic observations across all discovered sessions:

```bash
PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.method5_stereo_refinement.cache_observations \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --roma-prediction-root "$IMEDPE_OUTPUT_ROOT/roma2a" \
  --roma-root "$ROMA_ROOT" \
  --torch-home "$ROMA_TORCH_HOME" \
  --device cuda \
  --max-source-points 2048 \
  --certainty-threshold 0.20 \
  --cycle-threshold-px 2 \
  --frames-per-session 30 \
  --tracks-per-frame 150 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5r/cache"
```

Run the frozen optimizer, apply the relaxed GT-free validity rule, and verify
the selected calibration export:

```bash
PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.method5_stereo_refinement.run_refinement \
  --cache-root "$IMEDPE_OUTPUT_ROOT/method5r/cache" \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --rotation-bound-deg 3 \
  --translation-bound-deg 5 \
  --rotation-prior 100 \
  --translation-prior 100 \
  --huber-delta-px 2 \
  --max-nfev 80 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5r/fits"

PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.method5_relaxed.apply_relaxed_rule \
  --strict-run "$IMEDPE_OUTPUT_ROOT/method5r/fits" \
  --original-calibration-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5r/selected"

PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.method5_relaxed.verify_exports \
  --original-root "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --selected-root "$IMEDPE_OUTPUT_ROOT/method5r/selected/calibration" \
  --relaxed-summary "$IMEDPE_OUTPUT_ROOT/method5r/selected/relaxed_summary.json" \
  --output "$IMEDPE_OUTPUT_ROOT/method5r/selected/verification.json"
```

Regenerate RoMa-2A with the selected calibration and replace only the 2A branch
of frozen Method 4A:

```bash
PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python -m \
  experiments.roma_2a.run_method2a_roma \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root "$IMEDPE_OUTPUT_ROOT/method5r/selected/calibration" \
  --roma-root "$ROMA_ROOT" \
  --torch-home "$ROMA_TORCH_HOME" \
  --device cuda \
  --max-source-points 2048 \
  --certainty-threshold 0.20 \
  --cycle-threshold-px 2 \
  --debug-frames 0 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5r/roma2a"

PYTHONPATH="$PWD/src:$PWD" uv run --project environments/roma python \
  scripts/run_method4a.py \
  --split train \
  --method1-root "$IMEDPE_OUTPUT_ROOT/method1" \
  --method2a-root "$IMEDPE_OUTPUT_ROOT/method5r/roma2a" \
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
  --output-root "$IMEDPE_OUTPUT_ROOT/method5r/roma2a4a"
```

For an unseen split, repeat original session calibration, RoMa-2A, correction
estimation, selection, and prediction from that split's images. Never reuse a
calibration from another physical session or bundle released TRAIN/TEST
calibration outputs in a submission.

## Method 6B — selected routed ensemble

Method 6B uses original Method 2A, original Method 2B, LoFTR-MV 2A, Method 1
VO, and the Method 1.5A rotation source. Run those methods first using the
commands above. The router consumes their saved inference diagnostics and does
not rerun image matching.

Build the supervised TRAIN router dataset. Ground truth is read here only to
construct TRAIN reliability targets:

```bash
uv run python -m experiments.method6_router_ensemble.build_dataset \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --expert 2a="$IMEDPE_OUTPUT_ROOT/method2a/pure" \
  --expert 2b="$IMEDPE_OUTPUT_ROOT/method2b" \
  --expert loftr="$IMEDPE_OUTPUT_ROOT/loftr_mv/method2a" \
  --method1-root "$IMEDPE_OUTPUT_ROOT/method1" \
  --rotation-root "$IMEDPE_OUTPUT_ROOT/method1_5" \
  --e1-calibration-diagnostics "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --e2-calibration-diagnostics "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --with-train-targets \
  --output-root "$IMEDPE_OUTPUT_ROOT/method6/dataset_train"
```

Reproduce the selected physical-session LOSO configuration, then fit one final
router on all TRAIN sessions:

```bash
uv run python -m experiments.method6_router_ensemble.evaluate_loso \
  --dataset-root "$IMEDPE_OUTPUT_ROOT/method6/dataset_train" \
  --model logistic \
  --feature-set r3 \
  --smoothing-alpha 1 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method6/loso_logistic_r3"

uv run python -m experiments.method6_router_ensemble.train_final \
  --dataset-root "$IMEDPE_OUTPUT_ROOT/method6/dataset_train" \
  --model logistic \
  --feature-set r3 \
  --smoothing-alpha 1 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method6/final_router"
```

For a target split, build inference-only features without
`--with-train-targets`, then apply either the newly trained router or the
bundled frozen router under `models/method6/`:

```bash
uv run python -m experiments.method6_router_ensemble.build_dataset \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split test \
  --expert 2a="$IMEDPE_OUTPUT_ROOT/method2a/pure" \
  --expert 2b="$IMEDPE_OUTPUT_ROOT/method2b" \
  --expert loftr="$IMEDPE_OUTPUT_ROOT/loftr_mv/method2a" \
  --method1-root "$IMEDPE_OUTPUT_ROOT/method1" \
  --rotation-root "$IMEDPE_OUTPUT_ROOT/method1_5" \
  --e1-calibration-diagnostics "$IMEDPE_OUTPUT_ROOT/calibration/e1" \
  --e2-calibration-diagnostics "$IMEDPE_OUTPUT_ROOT/calibration/e2" \
  --output-root "$IMEDPE_OUTPUT_ROOT/method6/dataset_test"

uv run python -m experiments.method6_router_ensemble.predict_final \
  --dataset-root "$IMEDPE_OUTPUT_ROOT/method6/dataset_test" \
  --router-root models/method6 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method6/method6b"
```

The frozen project artifact contains no images or poses. `router.pkl` is the
exact scikit-learn 1.7.2 artifact used by the research CLI; `router.npz` and
`router.json` provide an equivalent, version-independent deployment export.
The frozen feature schema, hashes, and ablations are documented in
[`METHOD6_RESULTS.md`](METHOD6_RESULTS.md).

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
