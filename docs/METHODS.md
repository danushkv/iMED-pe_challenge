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

