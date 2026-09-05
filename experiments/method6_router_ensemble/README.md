# Method 6 — reliability-routed mixture of pose experts

Method 6 uses saved Method-2A, Method-2B, LoFTR-2A, and Method-1 outputs. It
does not rerun matching during router development. All absolute trajectories
are robustly aligned from predictions only into the Method-2B working frame.

The router receives inference-time PnP/calibration diagnostics (R1),
cross-expert disagreement and Method-1 innovation (R2), or their union (R3).
No sequence name, session ID, motion label, hidden score, or GT quantity is a
router feature.

TRAIN ground truth is used only to create trajectory-level expert-error labels.
Every reported router prediction is physical-session LOSO: the held session is
excluded from scaler fitting and model training. Final TEST inference uses a
router fitted once on all TRAIN sessions and never reads TEST `pose.txt`.

The first mandatory gate is the hard/convex oracle. If the oracle does not
meaningfully beat 4A-LoFTR, stop before router training.

