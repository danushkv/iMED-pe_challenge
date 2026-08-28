# RoMa runtime environment

RoMa was evaluated in a separate Python 3.10 environment because its official
full model and fused local-correlation extension require a matched PyTorch,
torchvision, and CUDA stack. The committed project and lockfile reproduce the
evaluated Python package graph.

```bash
uv sync --project environments/roma --frozen
bash third_party/setup_models.sh --roma
uv run --project environments/roma python scripts/check_install.py --require-roma
```

The evaluated system used PyTorch 2.11, torchvision 0.26, CUDA 13.0, and
`fused-local-corr` 0.3.211. A deployment image may use another compatible CUDA
build only after a numerical regression against the frozen predictions.

Weights are not committed. Place the official files at the paths documented in
`third_party/manifest.yaml`; the setup helper verifies their SHA-256 hashes.

