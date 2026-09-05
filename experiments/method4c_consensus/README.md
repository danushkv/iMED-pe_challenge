# Method 4C — multi-observer consensus + Stereo-VO

This experiment fuses saved absolute trajectories only. It never reruns ALIKED,
LightGlue, LoFTR, RoMa, Method 1, or Method 2.

Primary observers:

```text
2a     outputs/method_2/full/pure
2b     outputs/method_2b/full
loftr  outputs/efficient_loftr_multiview/full_train/method2a_epsilon3
roma   outputs/roma_2a/method2a_train_r2
```

TRAIN temporal/rotation inputs:

```text
Method 1 VO     outputs/stereo_vo_full
rotation        outputs/stereo_vo_full
```

The frozen Method-4A numeric settings are retained: `lambda_vo=5`, Huber step
scale `2.5`, eight alignment iterations, alignment MAD factor `2.5`, and at
most 200 optimizer evaluations.

## Staged protocol

### 1. Exact Method-4A regression

Run Method 4C with only 2A/2B and the special regression weighting:

```bash
PYTHONPATH="$PWD/src:$PWD" \
python -m experiments.method4c_consensus.run_method4c \
  --split train \
  --observer 2a=outputs/method_2/full/pure \
  --observer 2b=outputs/method_2b/full \
  --reference-observer 2b \
  --method1-root outputs/stereo_vo_full \
  --rotation-root outputs/stereo_vo_full \
  --mode 4c1 \
  --weighting method4a-v1 \
  --lambda-vo 5 \
  --sequence session_004_scene_1_circular \
  --output-root outputs/method4c_consensus/sanity/regression_method4a
```

Compare with frozen Method 4A:

```bash
PYTHONPATH="$PWD/src:$PWD" \
python -m experiments.method4c_consensus.verify_regression \
  --split train \
  --reference-root outputs/method_4a/train_full/method4a1_lambda5 \
  --candidate-root outputs/method4c_consensus/sanity/regression_method4a \
  --sequence session_004_scene_1_circular \
  --tolerance 1e-8 \
  --output outputs/method4c_consensus/sanity/regression_method4a.json
```

### 2. Required optimizer limit checks

Use the four primary observers on one sequence. With `lambda_vo=0`, the output
is the weighted robust absolute consensus and `lambda_zero_max_error` must be
zero. With `absolute_weight_scale=0`, the output is exact aligned VO integration
and `no_absolute_max_error` must be zero.

### 3. Three-sequence screen

Run 4C-0, 4C-1-V0, and 4C-1-V1 on:

```text
session_004_scene_1_circular
session_005_scene_1_left_right
session_007_scene_3_zoom_in
```

No parameters are selected from TEST or hidden scores.

### 4. Full TRAIN

If sanity passes, omit all `--sequence` arguments to process all 61 TRAIN
sequences. Evaluate ordinary metrics, session-balanced metrics, worst session,
CV, and aligned per-frame p95 error.

### 5. Leave-one-observer-out

Use the same V1 command and omit exactly one `--observer` argument for each of:

```text
drop 2a
drop 2b
drop loftr
drop roma
```

Do not enumerate every possible subset.

### 6. Freeze, then TEST once

Choose observer set and weighting using TRAIN robustness only. Replace the
roots with their released-TEST equivalents, use the existing frozen TEST
rotation source, and evaluate exactly once. Do not retune afterward.

## TRAIN-only complementarity diagnostic

`complementarity.py` reads GT only after predictions exist. It reports aligned
per-frame error correlations, conditional worst-decile overlap, and whether the
other observers recover one observer's bad frames. Its output is diagnostic and
is never consumed by Method 4C inference.

## Runtime

Offline consensus optimization is normally a fraction of a second per
sequence. Docker feasibility depends on the selected observer subset. Full
RoMa, LoFTR, original Method 2B, and Method 1 cannot be assumed to fit together;
use leave-one-observer-out TRAIN results before selecting a deployable subset.
