# Method 5 runbook

All calibration choices are made on TRAIN. The initial sanity sessions are:

- `session_004`: previously stable calibration;
- `session_005`: previously unstable calibration;
- `session_002`: difficult, high-ATE session.

The observation cache uses the frozen RoMa R2 configuration: 2048 ALIKED
source points, certainty at least 0.20, joint cycle error at most 2 px, 30
uniform frames per physical session, and at most 150 confidence-ranked tracks
per frame. It reads saved raw RoMa poses but never reads ground truth.

## 1. Cache sanity-session observations

```bash
envs/roma/bin/python -m experiments.method5_stereo_refinement.cache_observations \
  --data-root "$IMEDPE_DATA_ROOT" \
  --split train \
  --calibration-root outputs/method_2/calibration_train \
  --roma-prediction-root outputs/roma_2a/method2a_train_r2 \
  --roma-root third_party/RoMa \
  --torch-home third_party/RoMa/.torch \
  --device cuda \
  --max-source-points 2048 \
  --certainty-threshold 0.20 \
  --cycle-threshold-px 2 \
  --frames-per-session 30 \
  --tracks-per-frame 150 \
  --session 004 --session 005 --session 002 \
  --output-root outputs/method5_stereo_refinement/sanity/cache
```

## 2. Estimate and validate corrections

```bash
envs/roma/bin/python -m experiments.method5_stereo_refinement.run_refinement \
  --cache-root outputs/method5_stereo_refinement/sanity/cache \
  --calibration-root outputs/method_2/calibration_train \
  --rotation-bound-deg 3 \
  --translation-bound-deg 5 \
  --rotation-prior 100 \
  --translation-prior 100 \
  --huber-delta-px 2 \
  --max-nfev 80 \
  --output-root outputs/method5_stereo_refinement/sanity/refined

envs/roma/bin/python -m experiments.method5_stereo_refinement.compare_calibrations \
  --summary outputs/method5_stereo_refinement/sanity/refined/refinement_summary.json

envs/roma/bin/python -m experiments.method5_stereo_refinement.verify_exports \
  --original-root outputs/method_2/calibration_train \
  --candidate-root outputs/method5_stereo_refinement/sanity/refined/calibration \
  --refinement-summary outputs/method5_stereo_refinement/sanity/refined/refinement_summary.json \
  --output outputs/method5_stereo_refinement/sanity/refined/export_verification.json
```

Further full-TRAIN, regeneration, frozen-4A, and released-TEST commands must be
run only after this geometry-only sanity report passes. No setting is to be
changed based on TEST or hidden scores.

## Recorded result

The sanity stage was completed and all three corrections were rejected by the
frozen geometry-only rules. Export verification passed. The experiment stopped
without full TRAIN or TEST evaluation. See `FINAL_REPORT.md`.
