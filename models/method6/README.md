# Frozen Method 6B router

This directory contains the small router trained on all 61 released TRAIN
sequences after physical-session LOSO model selection.

| File | Purpose | SHA-256 |
|---|---|---|
| `router.pkl` | Exact scikit-learn 1.7.2 research artifact used by `predict_final.py` | `7b4c167d762e32f809455cd735c8ec40d31e7f4cbb0d5e7e4fb7545687298da2` |
| `router.npz` | Version-independent coefficients, intercept, classes, and robust scaler | `8841893c2c5096cb8287de8ccb67b49a632889a361de55ac53199a8e7b207254` |
| `router.json` | Feature ordering and export metadata | `73b0d44ce02bf7beb61ab859449769e1826300b6a5fa0334f75eda45306e723f` |

The artifacts contain no images, poses, ground truth, or per-sequence features.
Do not load pickle files obtained from untrusted sources. For deployment, prefer
the NPZ/JSON representation and the NumPy softmax implementation described in
`docs/METHOD6_RESULTS.md`.
