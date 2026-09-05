# Dataset contract

The code does not contain a dataset location. Every command receives
`--data-root`; examples use `$IMEDPE_DATA_ROOT`.

## Layout

```text
<DATA_ROOT>/<split>/<sequence>/
  K.txt
  pose.txt                         # present only where GT is released
  endoscope1/L/frame_XXXXXX.png
  endoscope1/R/frame_XXXXXX.png
  endoscope2/L/frame_XXXXXX.png
  endoscope2/R/frame_XXXXXX.png
```

`K.txt` must contain four named 3x3 matrices: `K1_L`, `K1_R`, `K2_L`, and
`K2_R`. All four streams must contain the same frame IDs.

Sequences are grouped by the `session_NNN_` prefix. Stereo extrinsics are not
assumed to be supplied. Methods 1 and 2 estimate one fixed, scale-free stereo
transform per physical session from synchronized images. This inference-time
self-calibration does not read `pose.txt`.

The evaluated dataset did not supply lens-distortion coefficients and did not
declare that images were rectified. Geometry code therefore uses the provided
intrinsics with `distCoeffs=None`; it does not claim to undistort the frames.

## Ground-truth isolation

Inference modules accept sequence image directories and calibration caches.
They do not require ground truth. `pose.txt` is read only by evaluation,
plotting, and dataset-validation utilities after predictions exist.

Do not commit raw dataset content. The challenge organizers permitted the two
curated qualitative examples under `assets/examples/`; this permission does not
grant general redistribution rights for the dataset. Obtain data and its
required citation from the
[official iMED Challenge website](https://imed-challenge.github.io/).
