# Method 5-R runbook

Method 5-Strict outputs remain untouched. Sessions 002, 004, and 005 reuse:

```text
outputs/method5_stereo_refinement/sanity/refined
```

## 1. Cache only the four remaining TRAIN sessions

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
  --session 001 --session 003 --session 006 --session 007 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5_relaxed/cache_remaining_train"
```

## 2. Run the unchanged strict optimizer for those fits

This generates fit diagnostics; its strict export decisions are not Method
5-R's decisions.

```bash
envs/roma/bin/python -m experiments.method5_stereo_refinement.run_refinement \
  --cache-root "$IMEDPE_OUTPUT_ROOT/method5_relaxed/cache_remaining_train" \
  --calibration-root outputs/method_2/calibration_train \
  --rotation-bound-deg 3 \
  --translation-bound-deg 5 \
  --rotation-prior 100 \
  --translation-prior 100 \
  --huber-delta-px 2 \
  --max-nfev 80 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5_relaxed/fits_remaining_train"
```

## 3. Apply the one global relaxed rule to all seven sessions

```bash
envs/roma/bin/python -m experiments.method5_relaxed.apply_relaxed_rule \
  --strict-run outputs/method5_stereo_refinement/sanity/refined \
  --strict-run "$IMEDPE_OUTPUT_ROOT/method5_relaxed/fits_remaining_train" \
  --original-calibration-root outputs/method_2/calibration_train \
  --require-session 001 --require-session 002 --require-session 003 \
  --require-session 004 --require-session 005 --require-session 006 \
  --require-session 007 \
  --output-root "$IMEDPE_OUTPUT_ROOT/method5_relaxed/selected_train"
```

Downstream regeneration commands are intentionally gated on inspection of the
seven-session `relaxed_summary.json`.
