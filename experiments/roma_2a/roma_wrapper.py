from __future__ import annotations

import hashlib
import importlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class DenseRoMaPrediction:
    warp: torch.Tensor
    certainty: torch.Tensor
    seconds: float
    peak_gpu_memory_gib: float
    reverse_warp: torch.Tensor | None = None
    reverse_certainty: torch.Tensor | None = None


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


class OfficialRoMa:
    """Thin, lazy wrapper around the official ``romatch`` implementation."""

    def __init__(
        self,
        roma_root: str | Path,
        device: str = "cuda",
        torch_home: str | Path | None = None,
    ) -> None:
        root = Path(roma_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Official RoMa checkout not found: {root}")
        if torch_home is not None:
            os.environ["TORCH_HOME"] = str(Path(torch_home).expanduser().resolve())
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            romatch = importlib.import_module("romatch")
        except ImportError as error:
            raise RuntimeError(
                "Could not import official romatch. Install the local RoMa checkout "
                "in the dedicated environment first."
            ) from error
        if not hasattr(romatch, "roma_outdoor"):
            raise RuntimeError("The installed romatch package has no roma_outdoor API")

        self.root = root
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        self.model = romatch.roma_outdoor(device=self.device)
        self.model.eval()
        self.symmetric = bool(getattr(self.model, "symmetric", False))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - started

    @torch.inference_mode()
    def match(self, source: str | Path, target: str | Path) -> DenseRoMaPrediction:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        warp, certainty = self.model.match(
            str(Path(source)),
            str(Path(target)),
            device=self.device,
        )
        # Full roma_outdoor currently performs symmetric inference.  Its public
        # ``match`` method concatenates A->B and B->A horizontally:
        #
        #   warp:      [batch, H, 2*W, 4]
        #   certainty: [batch, H, 2*W]
        #
        # The first W columns are the requested source(A)->target(B) map:
        #
        #   [A source coordinates, B target coordinates]
        #
        # The second W columns encode the simultaneously predicted reverse:
        #
        #   [A target coordinates, B source coordinates]
        #
        # Canonicalize the latter to [B source, A target] so it can be sampled
        # by the same ``sample_dense_warp`` helper.  This exposes both
        # directions from one official symmetric inference and avoids a
        # redundant explicit match(B, A) call during cycle checking.
        reverse_warp = None
        reverse_certainty = None
        if self.symmetric:
            warp_width = int(warp.shape[-2])
            certainty_width = int(certainty.shape[-1])
            if warp_width != certainty_width or warp_width % 2:
                raise RuntimeError(
                    "Unexpected symmetric RoMa output shapes: "
                    f"warp={tuple(warp.shape)}, certainty={tuple(certainty.shape)}"
                )
            forward_width = warp_width // 2
            reverse_raw = warp[..., forward_width:, :]
            reverse_warp = torch.cat(
                (reverse_raw[..., 2:4], reverse_raw[..., 0:2]), dim=-1
            )
            reverse_certainty = certainty[..., forward_width:]
            warp = warp[..., :forward_width, :]
            certainty = certainty[..., :forward_width]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device) / (1024.0**3)
        else:
            peak = 0.0
        return DenseRoMaPrediction(
            warp=warp,
            certainty=certainty,
            seconds=time.perf_counter() - started,
            peak_gpu_memory_gib=float(peak),
            reverse_warp=reverse_warp,
            reverse_certainty=reverse_certainty,
        )

    def metadata(self) -> dict[str, Any]:
        commit = None
        head = self.root / ".git" / "HEAD"
        if head.is_file():
            head_text = head.read_text().strip()
            if head_text.startswith("ref: "):
                ref = self.root / ".git" / head_text[5:]
                if ref.is_file():
                    commit = ref.read_text().strip()
            elif len(head_text) >= 7:
                commit = head_text
        return {
            "repository": "https://github.com/Parskatt/RoMa",
            "checkout": str(self.root),
            "commit": commit,
            "model": "roma_outdoor",
            "official_symmetric_inference": self.symmetric,
            "returned_dense_field": (
                "canonical forward A-to-B and reverse B-to-A fields from one "
                "official symmetric inference"
                if self.symmetric
                else "forward A-to-B field only"
            ),
            "device": str(self.device),
            "model_load_seconds": self.load_seconds,
            "torch_home": os.environ.get("TORCH_HOME"),
            "coordinate_convention": {
                "dense_warp": "H x W x 4 normalized coordinates",
                "source_channels": [0, 1],
                "target_channels": [2, 3],
                "normalized_range": [-1.0, 1.0],
                "pixel_to_normalized": "2 * pixel / image_size - 1",
                "sampling_align_corners": False,
            },
        }
