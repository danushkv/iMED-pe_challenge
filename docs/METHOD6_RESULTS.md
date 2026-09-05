# Method 6B results and frozen decision

## Frozen method

Method 6B is a soft reliability-routed mixture of three absolute-pose experts:

```text
original Method 2A
original Method 2B
EfficientLoFTR Method 2A
        +
original Method 1 Stereo-VO increments
        ->
frozen Method-4A-style robust trajectory optimizer
```

The selected router is multinomial logistic regression with the R3 union of
per-expert diagnostics, cross-expert disagreement, and VO-innovation features.
It uses robust median/MAD scaling, soft TRAIN error targets, no probability
smoothing (`alpha=1.0`), and no entropy gate. Selection used physical-session
LOSO only. Released TEST and hidden results did not select any setting.

## Oracle diagnostic

| Predictor | Overall ATE | Session-balanced ATE | Worst-session ATE | Frame p95 |
|---|---:|---:|---:|---:|
| Always 2A | 1.2002 | 1.7231 | 3.8866 | 3.3049 |
| Always 2B | 1.0996 | 1.4591 | 3.2645 | 3.0178 |
| Always LoFTR-2A | 1.3622 | 1.9242 | 4.6149 | 3.5014 |
| Hard per-frame oracle | 1.0713 | 1.4755 | 3.0425 | 3.0327 |
| Soft convex oracle | 0.7839 | 0.8936 | 1.6052 | 2.3842 |

Hard selection was not promising. The soft oracle showed enough headroom to
justify soft routing and VO-constrained fusion.

## TRAIN physical-session LOSO

| Method | Overall ATE | Session-balanced ATE | Worst-session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original Method 4A-1 | 0.8801 | 0.9576 | 1.5771 | 0.3504 | 2.6244 |
| Method 4A-1 + LoFTR-2A | 0.8968 | 0.9576 | 1.4647 | 0.3193 | 2.6252 |
| **Method 6B, logistic R3** | **0.8707** | **0.9156** | **1.4191** | 0.3244 | **2.5914** |

Logistic R3 narrowly outperformed logistic R1 under the predefined priority.
Gradient-boosted trees generalized less reliably. Entropy-gated Method 6C and
Method 6A without VO were rejected.

EMA smoothing was tested only at `alpha={0.3,0.5,0.7}` after selecting the
router. No smoothing remained best: it had the lowest session-balanced ATE,
ordinary ATE, and frame p95. Alpha 0.7 improved worst-session ATE by only
0.0009 mm while worsening the higher-priority aggregate metrics.

## Frozen released TEST evaluation

| Method | Overall ATE | Session-balanced ATE | Worst-session ATE | CV | Frame p95 |
|---|---:|---:|---:|---:|---:|
| Original Method 4A-1 | 1.1546 | 1.1889 | 2.0488 | 0.4596 | 3.1710 |
| Method 4A-1 + LoFTR-2A | 1.1402 | 1.1715 | 2.0349 | 0.4636 | 3.1481 |
| Method 4A-1 + RoMa-2A | 1.1465 | 1.1806 | 2.0371 | 0.4608 | 3.2105 |
| Method 5R + Method 4A-1 + RoMa-2A | 1.1447 | 1.1787 | 2.0371 | 0.4629 | 3.2129 |
| **Method 6B** | **1.1357** | **1.1663** | **2.0296** | 0.4645 | **3.0973** |

Per-session Method 6B ATE:

| Session | Sequences | Mean ATE |
|---|---:|---:|
| 002 | 5 | 0.7074 |
| 004 | 6 | 0.7031 |
| 005 | 3 | 1.2251 |
| 007 | 5 | 2.0296 |

The improvement over 4A-LoFTR is small but consistent across all prioritized
released-TEST metrics. Method 6B therefore qualifies for one frozen hidden
submission, subject to integrated Docker runtime validation.

## Runtime and artifact

For 1,895 released-TEST frames:

```text
router prediction                 0.0078 ms/frame
probability processing            0.0079 ms/frame
routed robust fusion              4.0897 ms/frame
total router/fusion postprocess  12.0736 s
```

The final router was trained on 61 TRAIN sequences, 6,095 frames, and seven
physical sessions with scikit-learn 1.7.2. The local artifact is 21,348 bytes:

```text
outputs/method6_router_ensemble/final_router/router.pkl
SHA256 7b4c167d762e32f809455cd735c8ec40d31e7f4cbb0d5e7e4fb7545687298da2
```

Raw predictions, evaluation JSON, and released dataset files are intentionally
excluded from the release repository.
