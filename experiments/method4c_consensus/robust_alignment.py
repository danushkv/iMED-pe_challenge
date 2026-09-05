from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from imcpe.methods.hybrid_pose_fusion.alignment import RobustSimilarity, robust_sim3

from .load_observers import LoadedTrajectory, ObserverBundle


@dataclass(frozen=True)
class AlignedObservers:
    names: tuple[str, ...]
    centers: np.ndarray  # M x N x 3
    valid: np.ndarray  # M x N
    similarities: dict[str, RobustSimilarity | None]
    reference_centers: np.ndarray
    trajectory_step_scale: float


def _step_scale(points: np.ndarray) -> float:
    displacement = np.linalg.norm(np.diff(points, axis=0), axis=1)
    nonzero = displacement[np.isfinite(displacement) & (displacement > 1.0e-9)]
    return float(np.median(nonzero)) if len(nonzero) else 1.0


def align_observers_to_reference(
    bundle: ObserverBundle,
    *,
    iterations: int,
    huber_mad: float,
) -> AlignedObservers:
    names = tuple(bundle.observers)
    reference = bundle.observers[bundle.reference_name]
    reference_centers = reference.centers - reference.centers[0]
    centers = np.full((len(names), len(bundle.frame_ids), 3), np.nan, dtype=np.float64)
    valid = np.zeros((len(names), len(bundle.frame_ids)), dtype=bool)
    similarities: dict[str, RobustSimilarity | None] = {}

    for observer_index, name in enumerate(names):
        trajectory: LoadedTrajectory = bundle.observers[name]
        mask = trajectory.valid & reference.valid
        if name == bundle.reference_name:
            centers[observer_index, trajectory.valid] = (
                trajectory.centers[trajectory.valid] - trajectory.centers[0]
            )
            valid[observer_index] = trajectory.valid
            similarities[name] = None
            continue
        if np.count_nonzero(mask) < 3:
            similarities[name] = None
            continue
        similarity = robust_sim3(
            trajectory.centers[mask],
            reference_centers[mask],
            iterations=iterations,
            huber_mad=huber_mad,
        )
        transformed = similarity.apply(trajectory.centers[trajectory.valid])
        if trajectory.valid[0]:
            transformed -= similarity.apply(trajectory.centers[[0]])[0]
        centers[observer_index, trajectory.valid] = transformed
        valid[observer_index] = trajectory.valid
        similarities[name] = similarity

    if not np.all(valid[names.index(bundle.reference_name)]):
        raise RuntimeError("Reference observer is incomplete")
    return AlignedObservers(
        names=names,
        centers=centers,
        valid=valid,
        similarities=similarities,
        reference_centers=reference_centers,
        trajectory_step_scale=_step_scale(reference_centers),
    )


def align_method1_to_reference(
    bundle: ObserverBundle,
    reference_centers: np.ndarray,
    *,
    iterations: int,
    huber_mad: float,
) -> RobustSimilarity:
    return robust_sim3(
        bundle.method1.centers,
        reference_centers,
        iterations=iterations,
        huber_mad=huber_mad,
    )

