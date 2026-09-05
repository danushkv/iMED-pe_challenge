from __future__ import annotations

import itertools

import numpy as np

from imcpe.methods.hybrid_pose_fusion.position_fusion import distribution

from .robust_alignment import AlignedObservers


def extended_distribution(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "maximum": float("nan"),
        }
    return {
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "maximum": float(np.max(finite)),
    }


def pairwise_disagreements(aligned: AlignedObservers) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    scale = max(aligned.trajectory_step_scale, 1.0e-9)
    for first, second in itertools.combinations(range(len(aligned.names)), 2):
        valid = aligned.valid[first] & aligned.valid[second]
        distance = np.linalg.norm(
            aligned.centers[first, valid] - aligned.centers[second, valid], axis=1
        )
        key = f"{aligned.names[first]}__{aligned.names[second]}"
        result[key] = {
            "num_common_frames": int(np.count_nonzero(valid)),
            "distance": extended_distribution(distance),
            "distance_normalized_by_step_scale": extended_distribution(distance / scale),
        }
    return result


def alignment_summary(
    aligned: AlignedObservers, reference_name: str
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in aligned.names:
        similarity = aligned.similarities[name]
        if similarity is None:
            result[name] = {
                "reference": name == reference_name,
                "scale": 1.0,
                "rotation": np.eye(3),
                "translation": np.zeros(3),
                "residual": distribution(np.zeros(1)),
            }
        else:
            result[name] = {
                "reference": False,
                "scale": similarity.scale,
                "rotation": similarity.rotation,
                "translation": similarity.translation,
                "residual": extended_distribution(similarity.residuals),
                "num_robustly_downweighted": int(
                    np.count_nonzero(similarity.weights < 0.999)
                ),
            }
    return result
