"""I/O for the compact, GT-free Method-5 observation cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SessionObservations:
    session_id: str
    sequences: np.ndarray
    frame_ids: np.ndarray
    temporal_fraction: np.ndarray
    offsets: np.ndarray
    x_e1l: np.ndarray
    x_e1r: np.ndarray
    x_e2l: np.ndarray
    confidence: np.ndarray
    K1_L: np.ndarray
    K1_R: np.ndarray
    K2_L: np.ndarray
    T_e2_from_e1_initial: np.ndarray
    original_R_LR: np.ndarray
    original_t_LR: np.ndarray
    suggested_min_depth: float
    suggested_max_depth: float

    @property
    def num_frames(self) -> int:
        return int(len(self.frame_ids))

    def frame_slice(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))

    def subset(self, indices: np.ndarray) -> "SessionObservations":
        chosen = np.asarray(indices, dtype=np.int64)
        point_indices: list[np.ndarray] = []
        offsets = [0]
        for index in chosen:
            values = np.arange(
                self.offsets[index], self.offsets[index + 1], dtype=np.int64
            )
            point_indices.append(values)
            offsets.append(offsets[-1] + len(values))
        points = (
            np.concatenate(point_indices)
            if point_indices
            else np.empty(0, dtype=np.int64)
        )
        return SessionObservations(
            session_id=self.session_id,
            sequences=self.sequences[chosen],
            frame_ids=self.frame_ids[chosen],
            temporal_fraction=self.temporal_fraction[chosen],
            offsets=np.asarray(offsets, dtype=np.int64),
            x_e1l=self.x_e1l[points],
            x_e1r=self.x_e1r[points],
            x_e2l=self.x_e2l[points],
            confidence=self.confidence[points],
            K1_L=self.K1_L[chosen],
            K1_R=self.K1_R[chosen],
            K2_L=self.K2_L[chosen],
            T_e2_from_e1_initial=self.T_e2_from_e1_initial[chosen],
            original_R_LR=self.original_R_LR,
            original_t_LR=self.original_t_LR,
            suggested_min_depth=self.suggested_min_depth,
            suggested_max_depth=self.suggested_max_depth,
        )


def load_session_observations(path: Path) -> SessionObservations:
    with np.load(path, allow_pickle=False) as archive:
        result = SessionObservations(
            session_id=str(archive["session_id"].item()).zfill(3),
            sequences=np.asarray(archive["sequences"]).astype(str),
            frame_ids=np.asarray(archive["frame_ids"], dtype=np.int64),
            temporal_fraction=np.asarray(
                archive["temporal_fraction"], dtype=np.float64
            ),
            offsets=np.asarray(archive["offsets"], dtype=np.int64),
            x_e1l=np.asarray(archive["x_e1l"], dtype=np.float64),
            x_e1r=np.asarray(archive["x_e1r"], dtype=np.float64),
            x_e2l=np.asarray(archive["x_e2l"], dtype=np.float64),
            confidence=np.asarray(archive["confidence"], dtype=np.float64),
            K1_L=np.asarray(archive["K1_L"], dtype=np.float64),
            K1_R=np.asarray(archive["K1_R"], dtype=np.float64),
            K2_L=np.asarray(archive["K2_L"], dtype=np.float64),
            T_e2_from_e1_initial=np.asarray(
                archive["T_e2_from_e1_initial"], dtype=np.float64
            ),
            original_R_LR=np.asarray(archive["original_R_LR"], dtype=np.float64),
            original_t_LR=np.asarray(archive["original_t_LR"], dtype=np.float64),
            suggested_min_depth=float(archive["suggested_min_depth"].item()),
            suggested_max_depth=float(archive["suggested_max_depth"].item()),
        )
    if len(result.offsets) != result.num_frames + 1:
        raise RuntimeError(f"Malformed frame offsets in {path}")
    if result.offsets[-1] != len(result.x_e1l):
        raise RuntimeError(f"Point count/offset mismatch in {path}")
    if not (
        result.x_e1l.shape == result.x_e1r.shape == result.x_e2l.shape
        and result.x_e1l.ndim == 2
        and result.x_e1l.shape[1] == 2
    ):
        raise RuntimeError(f"Malformed three-view coordinates in {path}")
    return result

