# Method 5-R — Stable-Correction Downstream Validation

## Scope

Method 5-R changes only the geometry-only validity rule used to accept a
session-level E1 stereo correction. The optimizer, RoMa observations,
calibration bounds, regularization, sampling, triangulation, PnP, and frozen
Method-4A settings are unchanged.

No ground truth is used to estimate or accept a correction. Ground truth is
loaded only by the evaluator after prediction.

## Validity rules

Method 5-Strict required convergence, bounded and stable corrections, an
improved total objective, and restrictive per-view reprojection checks.

Method 5-R accepts a correction when all of the following inference-time
conditions hold:

1. the optimizer converged;
2. the correction is inside the existing hard bounds;
3. the existing split-half stability check passes;
4. the existing cross-sequence stability check passes where applicable; and
5. the final total robust objective is lower than the original-calibration
   objective.

Per-view reprojection statistics remain diagnostics and are not vetoes.

## TRAIN correction selection

Strict accepted 2 sessions, while Relaxed accepted 5 sessions. The three
sessions rejected by Strict but accepted by Relaxed were:

- session_002
- session_004
- session_005

Sessions 001 and 003 retained the original calibration because optimization did
not converge. Sessions 006 and 007 passed both rules.

## TRAIN — RoMa-2A observations

| Method | Overall ATE | Session-balanced ATE | Worst session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original RoMa-2A | 1.0057 | 1.3025 | 3.1589 (002) | 0.6134 | 2.9139 |
| Method 5R RoMa-2A | 1.0002 | 1.3099 | 3.2603 (002) | 0.6401 | 2.9373 |

The standalone branch improved ordinary ATE but slightly worsened the robust
session-balanced, worst-session, CV, and tail metrics. It is not the primary
submission candidate.

Per-session original to corrected RoMa-2A ATE:

| Session | Original | Method 5-R | Change |
|---|---:|---:|---:|
| 001 | 1.0571 | 1.0571 | fallback/no change |
| 002 | 3.1589 | 3.2603 | +0.1014 |
| 003 | 1.3416 | 1.3416 | fallback/no change |
| 004 | 0.6707 | 0.6667 | -0.0040 |
| 005 | 0.9629 | 0.9347 | -0.0282 |
| 006 | 0.6347 | 0.6040 | -0.0307 |
| 007 | 1.2915 | 1.3048 | +0.0133 |

## TRAIN — frozen Method 4A with RoMa-2A

| Method | Overall ATE | Session-balanced ATE | Worst session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original Method 4A | 0.8801 | 0.9576 | 1.5771 (002) | 0.3504 | 2.6244 |
| Method 4A + XFeat | 0.8664 | 0.9018 | 1.2796 (007) | 0.2771 | 2.5826 |
| Method 4A + LoFTR-2A | 0.8968 | 0.9576 | 1.4647 (002) | 0.3193 | 2.6252 |
| Method 4A + RoMa-2A | 0.8477 | 0.8861 | 1.2388 (007) | 0.2737 | 2.5425 |
| **Method 5R + frozen 4A + RoMa-2A** | **0.8444** | **0.8777** | **1.2356 (007)** | **0.2655** | **2.5376** |

Method 5-R improved every selected TRAIN robustness metric over the original
RoMa-2A fusion without retuning Method 4A.

## LoFTR cross-check

Applying the same corrections to LoFTR did not generalize:

| Method | Overall ATE | Session-balanced ATE | Worst session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original LoFTR-2A + 4A | 0.8968 | 0.9576 | 1.4647 (002) | 0.3193 | 2.6252 |
| Method 5R LoFTR-2A + 4A | 0.9043 | 0.9824 | 1.6296 (002) | 0.3548 | 2.6834 |

The LoFTR variant is rejected. The selected candidate remains the RoMa branch.

## Released TEST — frozen evaluation

The TRAIN-selected rule was frozen before TEST. For TEST sessions, the same
inference-time optimizer and validity rule estimated corrections from images
and geometry only. Sessions 002 and 004 were accepted; sessions 005 and 007
fell back to their original calibration.

| Method | Overall ATE | Session-balanced ATE | Worst session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original RoMa-2A | 1.3089 | 1.4205 | 2.1265 (005) | 0.4839 | 3.4409 |
| Method 5R RoMa-2A | 1.3016 | 1.4137 | 2.1265 (005) | 0.4910 | 3.4385 |
| Original Method 4A | 1.1546 | 1.1889 | 2.0488 (007) | 0.4596 | 3.1710 |
| Original RoMa-2A + 4A | 1.1465 | 1.1806 | 2.0371 (007) | 0.4608 | 3.2105 |
| **Method 5R + frozen 4A + RoMa-2A** | **1.1447** | **1.1787** | **2.0371 (007)** | 0.4629 | 3.2129 |

TEST per-session original to corrected RoMa-2A ATE:

| Session | Original | Method 5-R | Result |
|---|---:|---:|---|
| 002 | 0.7533 | 0.7283 | improved |
| 004 | 0.7134 | 0.7111 | improved |
| 005 | 2.1265 | 2.1265 | fallback |
| 007 | 2.0887 | 2.0887 | fallback |

TEST per-session original to corrected frozen-fusion ATE:

| Session | Original RoMa-4A | Method 5R RoMa-4A | Result |
|---|---:|---:|---|
| 002 | 0.7161 | 0.7057 | improved |
| 004 | 0.7037 | 0.7065 | slight regression |
| 005 | 1.2654 | 1.2654 | fallback |
| 007 | 2.0371 | 2.0371 | fallback |

## Interpretation of the relaxed gate

1. Strict rejected three TRAIN sessions that Relaxed accepted: 002, 004, and
   005.
2. Those corrections were not uniformly beneficial frame-by-frame or
   session-by-session. Nevertheless, the final frozen RoMa-4A trajectory
   improved all selected aggregate TRAIN metrics.
3. The strict per-view reprojection veto was not a reliable proxy for downstream
   ATE. Session 005 had mixed per-view reprojection behavior but improved as a
   standalone corrected RoMa-2A branch. Sessions 002 and 004 were rejected by
   Strict, yet both improved in frozen fusion on TRAIN.
4. Mixed per-view reprojection behavior therefore does not necessarily imply
   worse pose ATE. The evidence is limited to seven TRAIN sessions, so this is a
   practical result rather than a broad statistical claim.

## Runtime and Docker feasibility

Measured released-TEST costs on the current system were approximately:

- calibration observation caching: 101 seconds for four physical sessions;
- calibration correction and stability fits: 92 seconds for four sessions;
- full RoMa-2A inference: approximately 83–86 seconds per sequence;
- frozen Method-4A fusion: approximately 4 seconds for 19 sequences.

Serial execution for all 19 released-TEST sequences would therefore be around
30 minutes before counting all original Method-1 and Method-2B work. This is not
compatible with a ten-minute submission as-is.

The hidden validation input reportedly has six sequences. At roughly 84 seconds
per sequence, RoMa alone is about 8.4 minutes. With one physical session,
calibration and fusion put the measured projection near 9–10 minutes before all
remaining branches; multiple physical sessions or duplicated image/model work
can exceed the limit. A Docker candidate must therefore share image decoding and
ALIKED work, reuse sampled RoMa observations in full inference, avoid duplicate
model calls, and be timed end-to-end on one RTX 4090 before submission.

Do not bundle TRAIN/TEST selected calibrations in the Docker. Hidden inference
must estimate the correction from hidden images with the frozen global rule and
fall back to the original calibration when invalid.

## Decision

Method 5R + frozen Method 4A + RoMa-2A is the only Method-5R candidate worth a
final hidden submission. It produced small but consistent aggregate improvements
on both TRAIN and released TEST. The improvement is modest, so submission is
recommended only if a complete offline Docker timing run safely satisfies the
ten-minute limit. Standalone Method 5R RoMa-2A and the corrected LoFTR variants
should not replace the fused candidate.

