# Docker

Method 6B is the selected final released-TEST candidate. Its small trained
router under `models/method6/` is the only TRAIN-derived artifact required at
inference; calibration and pose observations are generated from input images.

Challenge submission images were developed separately for Method 4A-1,
Method 4A-1 + XFeat calibration, and Method 4A-1 + EfficientLoFTR-MV.

They are not copied into this research repository because the challenge entry
points duplicate method code and bundle third-party assets under their own
licenses. A reproducible public Dockerfile should build from this repository
root and fetch the exact revisions in `third_party/manifest.yaml`; do not
commit image layers or third-party model weights.

The evaluated classical container basis was CUDA 11.8 / Ubuntu 22.04 / Python
3.10 with the package versions pinned in the root `pyproject.toml` and
`uv.lock`.
