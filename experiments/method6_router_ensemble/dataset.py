"""Portable per-sequence dataset reader used by oracle, LOSO, and final routing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SequenceRecord:
    sequence: str
    session: str
    split: str
    frame_ids: np.ndarray
    features: np.ndarray
    expert_centers: np.ndarray
    expert_valid: np.ndarray
    rotation_source: np.ndarray
    delta_vo: np.ndarray
    vo_weights: np.ndarray
    trajectory_step_scale: float
    expert_errors: np.ndarray | None
    best_expert: np.ndarray | None
    gt_centers: np.ndarray | None
    expert_common_centers_in_gt: np.ndarray | None


@dataclass(frozen=True)
class RouterDataset:
    root: Path
    split: str
    feature_names: tuple[str, ...]
    feature_groups: tuple[str, ...]
    records: tuple[SequenceRecord, ...]

    @property
    def sessions(self) -> tuple[str, ...]:
        return tuple(sorted({record.session for record in self.records}))


def load_dataset(root: Path) -> RouterDataset:
    manifest = json.loads((root / "manifest.json").read_text())
    records: list[SequenceRecord] = []
    for item in manifest["sequences"]:
        payload = np.load(root / item["npz"], allow_pickle=False)
        has_targets = bool(item["has_targets"])
        records.append(
            SequenceRecord(
                sequence=item["sequence"],
                session=item["session"],
                split=manifest["split"],
                frame_ids=payload["frame_ids"],
                features=payload["features"],
                expert_centers=payload["expert_centers"],
                expert_valid=payload["expert_valid"].astype(bool),
                rotation_source=payload["rotation_source"],
                delta_vo=payload["delta_vo"],
                vo_weights=payload["vo_weights"],
                trajectory_step_scale=float(payload["trajectory_step_scale"]),
                expert_errors=payload["expert_errors"] if has_targets else None,
                best_expert=payload["best_expert"] if has_targets else None,
                gt_centers=payload["gt_centers"] if has_targets else None,
                expert_common_centers_in_gt=(
                    payload["expert_common_centers_in_gt"] if has_targets else None
                ),
            )
        )
    return RouterDataset(
        root=root,
        split=manifest["split"],
        feature_names=tuple(manifest["feature_schema"]["names"]),
        feature_groups=tuple(manifest["feature_schema"]["groups"]),
        records=tuple(records),
    )


def feature_indices(dataset: RouterDataset, feature_set: str) -> np.ndarray:
    if feature_set == "r1":
        allowed = {"r1"}
    elif feature_set == "r2":
        allowed = {"r2"}
    elif feature_set == "r3":
        allowed = {"r1", "r2"}
    else:
        raise ValueError(feature_set)
    return np.asarray(
        [index for index, group in enumerate(dataset.feature_groups) if group in allowed],
        dtype=np.int64,
    )


def stack_records(
    records: list[SequenceRecord] | tuple[SequenceRecord, ...],
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.concatenate([record.features[:, indices] for record in records])
    if any(record.expert_errors is None for record in records):
        raise RuntimeError("Router training requested on a target-free dataset")
    errors = np.concatenate([record.expert_errors for record in records])
    return features, errors

