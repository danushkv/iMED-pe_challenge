from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Method3Config:
    """Inference configuration shared by Method 3A and Method 3B."""

    variant: str = "3a"
    preprocess_mode: str = "crop"
    dtype: str = "auto"
    minimum_stereo_baseline: float = 1.0e-6
    maximum_rotation_orthogonality_error: float = 1.0e-3
    minimum_rotation_determinant: float = 0.999
    maximum_rotation_determinant: float = 1.001
    failure_policy: str = "previous"

    def __post_init__(self) -> None:
        if self.variant not in {"3a", "3b"}:
            raise ValueError("variant must be '3a' or '3b'")
        if self.preprocess_mode not in {"crop", "pad"}:
            raise ValueError("preprocess_mode must be 'crop' or 'pad'")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("unsupported dtype")
        if self.failure_policy not in {"previous", "error"}:
            raise ValueError("failure_policy must be 'previous' or 'error'")
