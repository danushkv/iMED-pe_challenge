# Environments

The root lockfile covers the classical ALIKED/LightGlue, XFeat, and
EfficientLoFTR experiments. `vggt/` and `roma/` are isolated because their
official model stacks require different PyTorch ecosystems. Always invoke
their commands with the corresponding `--project` option.

This project records two distinct environments:

1. The root `pyproject.toml` and `uv.lock` reproduce the `pe-basline` research
   environment used for the local experiment suite.
2. `environments/docker/pyproject.toml` records the smaller CUDA 11.8 stack
   used by the challenge Method-4 Docker submissions.

They are intentionally separate. Package upgrades can produce small numerical
changes in RANSAC, triangulation, and nonlinear optimization, so reports should
state which environment was used.

VGGT has its own environment because it is a large zero-shot dependency and is
not needed by the classical methods.
