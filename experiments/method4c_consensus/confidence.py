from __future__ import annotations

import numpy as np

from imcpe.methods.hybrid_pose_fusion.confidence import build_anchor_confidence

from .consensus import ConsensusResult, ConsensusWeights, build_consensus_weights
from .load_observers import ObserverBundle
from .robust_alignment import AlignedObservers


def observer_weights(
    *,
    bundle: ObserverBundle,
    aligned: AlignedObservers,
    consensus: ConsensusResult,
    version: str,
) -> ConsensusWeights:
    if version in {"v0", "v1"}:
        return build_consensus_weights(consensus, aligned.valid, version=version)
    if version != "method4a-v1":
        raise ValueError(f"Unknown observer weighting {version}")
    if set(aligned.names) != {"2a", "2b"} or bundle.reference_name != "2b":
        raise RuntimeError(
            "method4a-v1 regression requires exactly observers 2a/2b and reference 2b"
        )
    index_a = aligned.names.index("2a")
    index_b = aligned.names.index("2b")
    trajectory_a = bundle.observers["2a"]
    trajectory_b = bundle.observers["2b"]
    confidence = build_anchor_confidence(
        frame_ids=bundle.frame_ids,
        centers_a_aligned=aligned.centers[index_a],
        centers_b=aligned.centers[index_b],
        rotations_a=trajectory_a.rotations_initial_from_camera,
        rotations_b=trajectory_b.rotations_initial_from_camera,
        frames_a=trajectory_a.frames,
        frames_b=trajectory_b.frames,
        summary_a=trajectory_a.summary,
        summary_b=trajectory_b.summary,
        version="v1",
    )
    weights = np.stack((confidence.weight_a, confidence.weight_b))
    # Preserve the caller's observer order.
    if aligned.names != ("2a", "2b"):
        weights = weights[[1, 0]]
    return ConsensusWeights(
        weights=weights,
        observer_robust_z=np.stack((confidence.robust_z, confidence.robust_z))[
            [0, 1] if aligned.names == ("2a", "2b") else [1, 0]
        ],
        frame_robust_z=confidence.robust_z,
        residual_median=float(np.median(confidence.position_disagreement)),
        residual_mad_sigma=max(
            1.4826
            * float(
                np.median(
                    np.abs(
                        confidence.position_disagreement
                        - np.median(confidence.position_disagreement)
                    )
                )
            ),
            1.0e-9,
        ),
        spread_median=float(np.median(consensus.spread)),
        spread_mad_sigma=max(
            1.4826
            * float(np.median(np.abs(consensus.spread - np.median(consensus.spread)))),
            1.0e-9,
        ),
    )

