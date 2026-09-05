# Method overview

## Method 1 — stereo visual odometry

For consecutive E2-L frames, the method intersects exact ALIKED keypoint IDs
participating in E2-L/E2-R stereo matching and E2-L temporal matching.
Session-fixed stereo geometry triangulates 3D points; PnP estimates each local
camera increment, which is accumulated from frame zero.

## Method 1.5A — rotation-anchored VO

Independent E1-L/E2-L essential matrices supply scale-independent rotation
anchors. Confidence-gated SO(3) interpolation corrects orientation periodically
while preserving and re-integrating Method 1 local translation increments.

## Method 2A/2B — independent stereo-PnP

Method 2A reconstructs E1 stereo geometry and solves E2-L PnP independently at
each timestamp. Method 2B reconstructs E2 stereo geometry, solves E1-L PnP,
and inverts the result. These methods avoid temporal accumulation but have
arbitrary session-consistent stereo scale.

## Method 3 — zero-shot VGGT

Method 3A processes each synchronized four-view timestamp independently.
Method 3B jointly processes the four reference views and four current views.
No challenge training or fine-tuning is performed.

## Method 4A — robust trajectory fusion

Method 4A robustly aligns Method 1 and Method 2A predicted camera-center
trajectories to Method 2B, uses A/B disagreement and PnP diagnostics as
inference-only confidence, and optimizes absolute anchors plus consecutive VO
displacements. No ground truth enters fusion.

## Front-end studies

Sparse XFeat was evaluated at 2048 features. It yielded more matches but lower
per-frame robust geometric purity. Its most useful configuration replaced only
E1 session stereo calibration in the Method 4A stack.

EfficientLoFTR produced strong pairwise cross-endoscope geometry. A first
ALIKED-to-LoFTR coordinate bridge left too few shared tracks; the subsequent
pure LoFTR multi-view variant used mutual nearest association between LoFTR
stereo and cross-camera source coordinates in original image pixels. The
selected setting used confidence 0.8 and a 3 px tolerance.

## RoMa-2A

RoMa-2A extracts up to 2048 ALIKED pixels once in E1-L and samples two dense
full-RoMa warps at those exact source coordinates: E1-L to E1-R and E1-L to
E2-L. This creates exact-index three-view tracks without nearest-neighbour
association between independently detected sets. The frozen configuration uses
certainty at least 0.20 and forward/backward cycle error at most 2 original
image pixels. Original Method-2A calibration, triangulation, PnP, refinement,
and frame-zero normalization remain unchanged.

## Method 5-R — stable session stereo correction

Method 5-R estimates a small six-parameter correction around the original E1
session stereo transform. It preserves baseline magnitude, bounds rotation to
3 degrees and translation-direction change to 5 degrees, and uses strong
priors. Each candidate is fitted from deterministic frames distributed across
the physical session and evaluated without ground truth.

A correction is accepted only when the optimizer converges, stays within
bounds, passes split-half and cross-sequence stability checks, and lowers the
total robust objective. Per-view reprojection statistics remain diagnostics;
they are not hard vetoes. Invalid corrections fall back to the original session
calibration. The selected trajectory replaces only the RoMa-2A observations in
the otherwise frozen Method-4A stack.

## Method 6B — reliability-routed mixture of pose experts

Method 6B uses original Method 2A, original Method 2B, and LoFTR-2A as
absolute-pose experts. A multinomial logistic router predicts a soft reliability
distribution per frame from inference-only PnP, calibration, cross-expert
disagreement, and Method-1 VO-innovation diagnostics. Physical-session LOSO is
used for router selection; no sequence or session identifier is a feature.

The frozen router uses the complete R3 feature set, robust median/MAD scaling,
soft TRAIN targets, no temporal probability smoothing, and no entropy gate.
Its weights replace the absolute-observation confidences in the unchanged
Method-4A VO-constrained optimizer. The rotation source, robust loss, VO
processing, and `lambda_vo=5` remain frozen.
