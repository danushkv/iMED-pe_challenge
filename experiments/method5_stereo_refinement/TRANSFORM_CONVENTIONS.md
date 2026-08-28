# Method 5 transform conventions

Method 5 reuses the conventions already verified by Method 2A and Method 4A.

`T_E1R_from_E1L` maps a point expressed in the E1-left camera to E1-right:

```text
X_E1R = R_LR X_E1L + t_LR
```

The E1-left projection matrix is `K1_L [I | 0]`; the E1-right projection
matrix is `K1_R [R_LR | t_LR]`. The original calibration stores a unit-length
translation because its baseline scale is arbitrary.

`T_E2_from_E1(t)` is the cross-endoscope pose returned by PnP:

```text
X_E2 = R_t X_E1 + t_t
```

Its camera center in the E1 reference frame is:

```text
C_t = -R_t.T @ t_t
```

Challenge predictions store the relative transform:

```text
T_E2(0)_from_E2(t) = T_cross(0) @ inverse(T_cross(t))
```

Consequently frame zero must be identity. This experiment never infers a
matrix order from ATE and never opens `pose.txt` during caching, calibration
refinement, or trajectory generation.

The corrected stereo transform is

```text
R'_LR = Exp(delta_theta) @ R_LR
d'_LR = Exp(delta_phi) @ normalize(t_LR)
t'_LR = ||t_LR|| d'_LR
```

Only direction, not baseline magnitude, is changed.
