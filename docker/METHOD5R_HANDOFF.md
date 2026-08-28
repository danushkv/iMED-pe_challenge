# Method 5-R RoMa/4A Docker handoff

## Candidate

Package only the selected final candidate:

```text
Method 5-R selected E1 calibration
  -> RoMa-2A absolute observations
  + original Method 2B
  + original Method 1 Stereo-VO
  -> frozen Method 4A-1
```

Do not package standalone corrected RoMa-2A or the corrected LoFTR cross-check
as the primary candidate.

## Runtime pipeline

1. Discover sequences and group them by physical session.
2. Estimate the original E1 stereo calibration from that session's images.
3. Deterministically sample 30 frames across the session.
4. Cache calibration observations with full RoMa and the frozen values in
   `configs/method5r_stereo_refinement.yaml`.
5. Run the unchanged Method-5 correction optimizer.
6. Apply the frozen Method-5R validity rule. If it fails, retain the original
   session calibration.
7. Generate RoMa-2A observations with the selected calibration.
8. Generate original Method-2B and Method-1 observations.
9. Run frozen Method 4A with only its 2A branch replaced.
10. Write one valid `pose.txt` for every input sequence with exactly the input
    frame IDs and count.

Do not bundle released TRAIN/TEST calibrations, observations, predictions, or
evaluation files. Hidden corrections are inference-time, image-only estimates.
Ground truth must never be opened.

## Frozen settings

Use these tracked files as the single source of truth:

- `configs/roma_2a.yaml`
- `configs/method5r_stereo_refinement.yaml`
- `configs/method4a_fusion.yaml`

In particular, Method 4A remains `4a1`, confidence `v1`, `lambda_vo=5`, unit
absolute/VO scales, Huber delta 2.5 steps, eight alignment iterations, alignment
MAD 2.5, and at most 200 optimizer evaluations.

## Offline model assets

Bundle the official sources and checkpoints described in
`third_party/manifest.yaml`, subject to their licenses. Verify all revision and
weight hashes during the build. Runtime must succeed with networking disabled.

## Runtime optimization boundaries

The research pipeline is close to the ten-minute limit for six sequences.
Optimize shared computation without changing numerical behavior:

- load RoMa only once;
- decode each image once;
- reuse ALIKED source features;
- reuse calibration-frame RoMa warps during full RoMa-2A prediction;
- avoid duplicate forward/reverse RoMa calls;
- share intrinsics and original session-calibration data;
- do not replace the model, thresholds, optimizer, PnP, or fusion settings.

Benchmark the complete container on one RTX 4090. Report model load, original
calibration, Method-5 caching/refinement, full RoMa-2A, Method 1/2B, fusion,
serialization, total wall time, and peak GPU memory separately.

## Required regression checks

1. Run the image with `--network=none`.
2. Confirm every input sequence produces a finite, valid trajectory with the
   correct frame IDs/count.
3. Confirm frame zero is normalized correctly.
4. Exercise both correction acceptance and original-calibration fallback.
5. Compare at least one sequence against the non-Docker pipeline using a strict
   numerical tolerance.
6. Run a complete representative six-sequence workload on one RTX 4090.
7. Treat exceeding ten minutes or requiring internet as a release blocker.

## Local regression targets

The frozen released-TEST evaluation was:

| Method | Overall ATE | Session-balanced ATE | Worst-session ATE |
|---|---:|---:|---:|
| Original RoMa-2A + Method 4A | 1.1465 | 1.1806 | 2.0371 |
| Method 5-R RoMa-2A + Method 4A | 1.1447 | 1.1787 | 2.0371 |

These values are regression references only. They must not be read by inference
or used to change the frozen validity rule.

## Docker-agent completion report

Return:

- files created;
- image tag and exact build command;
- exact offline run command;
- source revisions and checkpoint hashes;
- numerical regression differences;
- per-sequence output validation;
- total one-RTX-4090 runtime and phase breakdown;
- peak GPU memory;
- whether the image has a safe margin below ten minutes;
- remaining submission risks.

