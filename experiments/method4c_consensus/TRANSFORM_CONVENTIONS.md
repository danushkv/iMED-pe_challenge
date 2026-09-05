# Method 4C transform conventions

## Repository `pose.txt`

The repository has already numerically verified that saved challenge rows are
the relative pose

```text
T_0_from_t = [R_0_from_t, C_t]
```

where the stored three-vector is the current camera center `C_t` expressed in
the frame-zero coordinate system. It is **not** the `t` term of a
camera-from-world matrix.

Consequently Method 4C loads camera centers as:

```python
C_t = pose_row.t
```

It must not apply `-R.T @ t` to saved challenge rows. That formula applies when
the stored matrix satisfies `X_camera = R X_world + t`; it does not apply to
this repository's already-converted `T_0_from_t` rows.

At frame zero every valid trajectory must satisfy:

```text
C_0 = [0, 0, 0]
R_0_from_0 = I
```

## Predicted-only observer alignment

Method 2B is the internal reference. For every other absolute observer `m`, a
robust predicted-only similarity is fitted:

```text
C_m_aligned = s_m Q_m C_m + u_m
```

using only corresponding saved prediction frames. No challenge ground truth is
opened. The aligned trajectory is rebased so its valid frame-zero center is
zero.

This similarity changes only the internal position coordinate system. Method
4C does not fuse or realign observer rotations.

## Stereo-VO edges

Method 1 stores consecutive local motion implicitly in its relative poses. The
same recovery used by Method 4A is retained:

```text
T_previous_from_current = inverse(T_0_from_previous) @ T_0_from_current
```

Its local translation is rotated into Method-1 frame zero and then transformed
as a vector by the predicted-only Method1-to-Method2B similarity:

```text
DeltaC_vo_aligned = s_vo Q_vo DeltaC_vo
```

Similarity translation is never applied to a displacement vector.

## Final pose output

The optimizer produces final camera centers `C_final(t)`. Rotations are copied
unchanged from the frozen Method-4A rotation source. Output rows are:

```text
translation field = C_final(t)
quaternion field  = quaternion(R_frozen(t))
```

This matches the verified repository convention. Frame zero is explicitly
checked, and all rotations are checked for orthogonality and determinant one.

## Ground-truth boundary

The following modules never open dataset `pose.txt` ground truth:

```text
load_observers.py
robust_alignment.py
consensus.py
confidence.py
optimize.py
run_method4c.py
```

Ground truth is opened only by post-prediction tools:

```text
evaluate.py
summarize.py (reads evaluation JSON only)
complementarity.py
```

