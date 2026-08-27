# Results

> Keep this document private until the challenge permits method/result release.

All values below are local evaluations with the challenge-compatible evaluator:
one Horn/Sim(3) alignment per sequence followed by ATE. `TRAIN` contains 61
released training sequences; released `TEST` contains 19 locally evaluable
sequences. These numbers are not official hidden-test scores.

## Mean ATE (mm)

| Method | TRAIN (61) | Released TEST (19) |
|---|---:|---:|
| Baseline | not recorded in the final 61-sequence comparison | 2.0421 |
| Method 1, ALIKED 2048 | 1.1580 | 1.5343 |
| Method 1.5A | not recorded on all 61 | 1.5292 |
| Method 2A | 1.2002 | 1.4000 |
| Method 2B | 1.0996 | 1.3373 |
| Method 3A | not run on all 61 | not run on all 19 |
| Method 3B | not run on all 61 | 1.6986 |
| Method 4A-0 | 1.0416 | 1.2916 |
| Method 4A-1 | 0.8801 | 1.1546 |
| Method 4A-1 + XFeat E1 calibration | **0.8664** | 1.1563 |
| Method 4A-1 + LoFTR-MV 2A | 0.8968 | **1.1402** |
| Method 4A-1 + LoFTR-MV 2B | 0.8930 | 1.1867 |
| Method 4A-1 + LoFTR-MV A+B | 0.9073 | 1.1710 |

Bold marks the best local result in each column, not a claim of unseen-session
superiority.

## Session-balanced validation

LOSO here groups predictions by physical acquisition session. It is a reporting
framework—not retraining—unless a method has candidate hyperparameters, in
which case only development sessions may select them.

### TRAIN: seven released sessions

| Method | Session-balanced ATE | Worst-session ATE |
|---|---:|---:|
| Method 2A | 1.7231 | 3.8866 |
| Method 2B | 1.4591 | 3.2645 |
| Method 4A-1 | 0.9576 | 1.5771 |
| Method 4A-1 + XFeat E1 calibration | **0.9018** | **1.2796** |
| Method 4A-1 + LoFTR-MV 2A | 0.9576 | 1.4647 |
| Method 4A-1 + LoFTR-MV 2B | 0.9837 | 1.5356 |
| Method 4A-1 + LoFTR-MV A+B | 0.9758 | 1.3726 |

### Released TEST: four sessions

| Method | Session-balanced ATE | Worst-session ATE |
|---|---:|---:|
| Baseline | 2.0431 | 2.5239 |
| Method 1 | 1.5594 | 2.3011 |
| Method 2A | 1.5220 | 2.2989 |
| Method 2B | 1.4115 | 2.1817 |
| Method 4A-1 | 1.1889 | 2.0488 |
| Method 4A-1 + XFeat E1 calibration | 1.1959 | 2.0391 |
| Method 4A-1 + LoFTR-MV 2A | **1.1715** | 2.0349 |
| Method 4A-1 + LoFTR-MV 2B | 1.2372 | 2.0335 |
| Method 4A-1 + LoFTR-MV A+B | 1.2172 | **2.0125** |

## Interpretation

- Method 4A consistently improved over its Method 1/2 components locally.
- XFeat's per-frame matching was not promoted; only its calibration-only hybrid
  survived full-TRAIN validation.
- EfficientLoFTR had stronger pairwise geometry but weaker standalone Method 2
  trajectories. Its 2A observations modestly improved local TEST Method 4A.
- TRAIN and released TEST disagree on the best front-end variant. This is why
  all variants and LOSO statistics are reported instead of presenting one
  cherry-picked number.

Raw prediction files and evaluation JSON are intentionally excluded. Recreate
them with the commands in `docs/RUN_METHODS.md`.

## Official hidden validation

These are challenge-server results and are reported separately from the local
TRAIN and released-TEST evaluations above. Lower is better.

| Submission | Hidden mean ATE RMSE (mm) |
|---|---:|
| Method 2A | 2.474 |
| Method 2B | 2.484 |
| Method 4A-1 | 2.316 |
| Method 4A-1 + XFeat E1 calibration | approximately 2.31* |
| **Method 4A-1 + LoFTR-MV 2A** | **2.308** |
| Method 4A-1 + LoFTR-MV A+B | 2.332 |

\*The exact XFeat leaderboard value was not preserved in the experiment notes;
replace this approximation if the exact score is recovered.

Among the exact recorded scores, Method 4A-1 + LoFTR-MV 2A is currently the
strongest hidden-validation submission. It improves on original Method 4A-1 by
0.008 mm and on the LoFTR A+B variant by 0.024 mm. Hidden scores were not used
to tune the LoFTR confidence threshold, association tolerance, or Method 4A
fusion settings.
