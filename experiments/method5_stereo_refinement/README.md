# Method 5 — session stereo refinement

This isolated experiment estimates a small E1 stereo correction from cached
RoMa three-view observations. It has three explicit stages:

1. cache a deterministic TRAIN-only set of RoMa tracks (the only neural work);
2. optimize and validate one six-parameter correction per physical session;
3. rerun the unchanged RoMa-2A geometry with the exported calibration files.

Corrections are rejected using reprojection, split-half, cross-sequence, bound,
and convergence checks only. Rejected sessions receive a byte-equivalent
geometric copy of their original calibration values. Ground truth is used only
by later evaluation commands.

See `TRANSFORM_CONVENTIONS.md` for all matrix conventions. Commands are
documented in `RUNBOOK.md` after the first sanity stage is completed.
