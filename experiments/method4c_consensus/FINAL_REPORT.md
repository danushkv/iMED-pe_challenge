# Method 4C — Multi-Observer Consensus + Stereo-VO Fusion

## Outcome

Method 4C did **not** improve TRAIN robustness over the strongest existing
Method 4A variants. In accordance with the predefined stop condition, no
released-TEST selection/tuning and no Docker integration were performed.

The strongest existing TRAIN reference remained Method 4A + RoMa-2A:

| Method | Overall ATE (mm) | Session-balanced ATE (mm) | Worst session ATE (mm) | CV | Frame p95 (mm) |
|---|---:|---:|---:|---:|---:|
| Original Method 4A | 0.8801 | 0.9576 | 1.5771 | 0.3504 | 2.6244 |
| Method 4A + XFeat | 0.8664 | 0.9018 | 1.2796 | 0.2771 | 2.5826 |
| Method 4A + LoFTR-2A | 0.8968 | 0.9576 | 1.4647 | 0.3193 | 2.6252 |
| **Method 4A + RoMa-2A** | **0.8477** | **0.8861** | **1.2388** | **0.2737** | **2.5425** |
| Method 4C-0 | 0.9960 | 1.2441 | 2.7946 | 0.5539 | 2.9324 |
| Method 4C-1-V0 | 0.8844 | 0.9910 | 1.8102 | 0.4045 | 2.6784 |
| Method 4C-1-V1 | 0.8834 | 1.0339 | 2.1598 | 0.4894 | 2.6541 |

## Correctness checks

- The generalized two-observer optimizer reproduced frozen Method 4A with
  exactly zero pose-component error.
- With `lambda_vo=0`, `lambda_zero_max_error=0.0`.
- With all absolute weights disabled, `no_absolute_max_error=0.0` and the VO
  residual was numerical noise (maximum approximately `1.01e-16`).
- Inference alignment, weighting, consensus, and optimization did not load GT.
- TRAIN GT was used only by the post-inference complementarity diagnostic.

## Three-sequence screen

The screen initially made consensus-aware V1 look promising:

| Method | Mean ATE (mm) | Aligned frame p95 (mm) |
|---|---:|---:|
| Original Method 4A | 0.7408 | 1.7332 |
| Method 4A + RoMa-2A | 0.7117 | 1.5935 |
| Method 4C-0 | 0.8613 | 1.6572 |
| Method 4C-1-V0 | 0.7257 | 1.5791 |
| Method 4C-1-V1 | 0.7098 | 1.5292 |

The apparent V1 gain did not survive all 61 TRAIN sequences. This reinforces
the need for full-session confirmation rather than selecting from a small
screen.

## Leave-one-observer-out ablation

| Observer set | Overall ATE (mm) | Session-balanced ATE (mm) | Worst session ATE (mm) | CV | Frame p95 (mm) |
|---|---:|---:|---:|---:|---:|
| Full four-observer V1 | 0.8834 | 1.0339 | 2.1598 | 0.4894 | 2.6541 |
| Drop original 2A | 0.8803 | 1.0226 | 2.0974 | 0.4740 | 2.6533 |
| Drop original 2B | 0.8822 | 1.0549 | 2.2770 | 0.5127 | 2.6618 |
| Drop LoFTR-2A | 0.8720 | 0.9955 | 1.9430 | 0.4395 | 2.6079 |
| Drop RoMa-2A | 0.8995 | 1.0306 | 2.0864 | 0.4659 | 2.7217 |

Dropping LoFTR was the least harmful subset, but it still remained clearly
worse than Method 4A + RoMa-2A (`0.8861` session-balanced and `1.2388` worst
session). Therefore no observer subset qualified for TEST.

## Observer complementarity

TRAIN aligned per-frame positional-error correlations:

| | Original 2A | Original 2B | LoFTR-2A | RoMa-2A |
|---|---:|---:|---:|---:|
| Original 2A | 1.0000 | 0.6764 | 0.7040 | 0.7190 |
| Original 2B | 0.6764 | 1.0000 | 0.6726 | 0.6574 |
| LoFTR-2A | 0.7040 | 0.6726 | 1.0000 | 0.6829 |
| RoMa-2A | 0.7190 | 0.6574 | 0.6829 | 1.0000 |

Worst-decile recoverability by the median of the other observers:

| Failing observer | Other-observer median at most 75% of own error | Median other/own error ratio |
|---|---:|---:|
| Original 2A | 32.6% | 0.881 |
| Original 2B | 22.3% | 0.950 |
| LoFTR-2A | 48.2% | 0.765 |
| RoMa-2A | 22.3% | 0.963 |

LoFTR has the most recoverable bad frames, which agrees with `drop LoFTR`
being the best leave-one-out result. RoMa and original 2B failures are rarely
substantially corrected by the remaining observers. The correlations of
approximately `0.66–0.72` show that errors are neither independent nor fully
identical: limited complementarity exists, but inference-time geometric-median
agreement does not reliably identify the correct observer.

## Interpretation

The remaining errors appear substantially systematic rather than isolated
matcher-specific outliers. Three Method-2A-style observers share stereo
calibration, triangulation, PnP, and normalization structure; correlated bias
can therefore form a false consensus. Robust loss suppresses a single distant
trajectory, but cannot distinguish a correlated majority from the correct
observer without an independent reliability signal.

No V2 diagnostic weighting was attempted because V1 failed the stated TRAIN
stop condition. No robust-loss constants, observer priors, or VO weights were
retuned. Hidden leaderboard results were not used for inference weighting or
selection.

## Decision

- Do not submit or Dockerize Method 4C.
- Do not run Method 4C on released TEST solely to seek a favorable result.
- Retain Method 4A + LoFTR-2A as the best current hidden submission (`2.308 mm`).
- Retain Method 4A + RoMa-2A as the strongest robust TRAIN configuration and a
  competitive hidden submission (`2.310 mm`).
- A future consensus attempt would require a genuinely independent confidence
  signal or geometry model, not more tuning of the same position-median scheme.

## Saved artifacts

- Exact regression: `outputs/method4c_consensus/sanity/regression_method4a.json`
- Full TRAIN comparison: `outputs/method4c_consensus/full_train/comparison.json`
- Full TRAIN robustness summary:
  `outputs/method4c_consensus/full_train/session_summary.json`
- Leave-one-out comparison:
  `outputs/method4c_consensus/leave_one_out/comparison.json`
- Leave-one-out robustness summary:
  `outputs/method4c_consensus/leave_one_out/session_summary.json`
- Complementarity diagnostic:
  `outputs/method4c_consensus/diagnostics/complementarity_train.json`

