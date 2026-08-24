# Docker

Challenge submission images were developed separately for Method 4A-1,
Method 4A-1 + XFeat calibration, and Method 4A-1 + EfficientLoFTR-MV.

They are not copied into this research repository because the challenge entry
points duplicate method code and bundle third-party assets under their own
licenses. After the embargo, add a Dockerfile that builds from this repository
root and fetches the exact revisions in `third_party/manifest.yaml`; do not
commit image layers or model weights.

The evaluated classical container basis was CUDA 11.8 / Ubuntu 22.04 / Python
3.10 with the package versions pinned in the root `pyproject.toml` and
`uv.lock`.

