# Transform conventions

`T_A_from_B` maps coordinates from frame `B` into frame `A`:

```text
X_A = T_A_from_B X_B
```

## Challenge pose files

Rows have the form:

```text
frame_id tx ty tz qx qy qz qw
```

The quaternion order is `xyzw`. The stored pose is `T_0_from_t`:

```text
X_0 = R_0_from_t X_t + C_t
```

Consequently, the stored translation is the current camera origin expressed in
frame zero—the camera center `C_t`. It is not a world-to-camera translation,
and must not be converted using `-R.T @ t`. Frame zero is identity.

## Method 1

Stereo triangulation creates points in the previous E2-L camera. OpenCV PnP
returns:

```text
Delta_T_vo = T_current_from_previous
```

The implementation inverts and accumulates this increment:

```text
T_0_from_current = T_0_from_previous @ inverse(T_current_from_previous)
```

## Method 1.5A

Cross-endoscope essential-matrix matching returns
`T_E2(t)_from_E1(t)` up to translation scale. The scale-independent rotation
anchor relative to frame zero is:

```text
R_anchor(t) = R_cross(0) @ R_cross(t).T
```

Rotation is corrected on SO(3), then subsequent local translations are
re-integrated using the corrected orientation. `alpha=0` reproduces Method 1.

## Method 2A

E1 stereo points are object points and E2-L pixels are image observations, so
OpenCV PnP returns:

```text
T_cross(t) = T_E2(t)_from_E1
```

The challenge-relative trajectory is:

```text
T_rel(t) = T_cross(0) @ inverse(T_cross(t))
```

## Method 2B

Reverse PnP first returns `U(t) = T_E1_from_E2(t)`. Therefore:

```text
T_rel(t) = inverse(U(0)) @ U(t)
```

The implementation verifies the raw inverse numerically.

## Method 3

VGGT predicts OpenCV camera-from-world extrinsics. For two cameras in the same
VGGT call:

```text
T_B_from_A = T_B_from_world @ inverse(T_A_from_world)
```

Global translation scale is normalized by the predicted E1 stereo camera-center
distance, never by cross-endoscope translation. Method 3B estimates reference
and current configurations jointly.

## Method 4A

All inputs are first represented by stored camera centers and `R_0_from_t`.
Method 2A and Method 1 centers are robustly Sim(3)-aligned to Method 2B using
predictions only. Optimized centers are written directly as pose translations;
final rotations come from Method 1.5A.

