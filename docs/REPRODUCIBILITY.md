# Reproducibility and provenance

## Source provenance

The release was assembled from the working challenge repository at Git commit:

```text
a2267625c51f52c5e14ec5f1cd0610ec57a44eaf
```

Canonical method code was copied from `src/imcpe`; XFeat, EfficientLoFTR, and
LOSO code came from their isolated experiment directories. Generated outputs,
third-party checkouts, checkpoints, dataset files, and Python caches were not
copied.

## Environments

- Root `pyproject.toml` + `uv.lock`: exact direct packages observed in the
  `pe-basline` research environment, with transitive resolution locked.
- `environments/docker/`: frozen CUDA 11.8 challenge-container stack.
- `environments/vggt/`: camera-only VGGT experiment stack.

Use `uv sync --frozen`; omitting `--frozen` permits dependency re-resolution
and is not recommended for result reproduction.

## Determinism boundaries

The code fixes method thresholds and third-party revisions, but GPU feature
matching, USAC/MAGSAC, PnP RANSAC, and nonlinear solvers may exhibit small
hardware/library-dependent numerical changes. Record:

```text
GPU model
driver version
CUDA runtime
uv.lock commit
method config
third-party source commit
checkpoint SHA-256
```

Session self-calibration is part of inference. Its outputs are caches, not
trained checkpoints. Recreate them from the same session images and fixed
configuration.

## Result protocol

1. Choose thresholds/hyperparameters on TRAIN only.
2. Freeze configs before released TEST evaluation.
3. Run inference without `pose.txt` access.
4. Evaluate afterward using `scripts/evaluate_ate.py`.
5. Report both ordinary sequence mean and session-balanced/LOSO summaries.

The Markdown tables in `docs/RESULTS.md` are transcribed summaries. They are
not used as program input. Raw JSON is deliberately excluded and can be
recreated with the documented evaluation commands.

## Fresh-clone checklist

```bash
uv sync --extra dev --frozen
bash third_party/setup_models.sh --xfeat
bash third_party/setup_models.sh --loftr
uv run --frozen python scripts/check_install.py --require-xfeat --require-loftr
uv run --frozen pytest
uv run --frozen python scripts/audit_release.py
```

No dataset is needed for these static/synthetic checks. Model inference begins
only when a user invokes one of the run scripts with `--data-root`.

