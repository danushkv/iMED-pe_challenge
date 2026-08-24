from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .cross_pnp import Method2DebugFrame


def _read_color(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def _draw_matches(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    label: str,
    inlier_mask: np.ndarray | None = None,
    max_draw: int = 160,
) -> np.ndarray:
    height = max(image0.shape[0], image1.shape[0])
    width0 = image0.shape[1]
    canvas = np.zeros((height, width0 + image1.shape[1], 3), dtype=np.uint8)
    canvas[: image0.shape[0], :width0] = image0
    canvas[: image1.shape[0], width0:] = image1
    count = int(points0.shape[0])
    indices = (
        np.linspace(0, count - 1, max_draw, dtype=int)
        if count > max_draw
        else np.arange(count)
    )
    for index in indices:
        is_inlier = inlier_mask is None or bool(inlier_mask[index])
        color = (0, 220, 0) if is_inlier else (0, 0, 255)
        point0_array = np.rint(points0[index]).astype(int)
        point1_array = np.rint(points1[index]).astype(int)
        point0 = (int(point0_array[0]), int(point0_array[1]))
        point1 = (int(point1_array[0] + width0), int(point1_array[1]))
        cv2.circle(canvas, point0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point1, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, point0, point1, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    suffix = " green=inlier red=outlier" if inlier_mask is not None else ""
    cv2.putText(
        canvas,
        f"{label}: {count} (showing {len(indices)}){suffix}",
        (10, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_method2_debug_image(debug: Method2DebugFrame, output_path: Path) -> None:
    e1_left = _read_color(debug.e1_left_path)
    e1_right = _read_color(debug.e1_right_path)
    e2_left = _read_color(debug.e2_left_path)
    stereo = _draw_matches(
        e1_left,
        e1_right,
        debug.stereo_e1_left_points,
        debug.stereo_e1_right_points,
        "A: E1-L <-> E1-R stereo",
    )
    cross = _draw_matches(
        e1_left,
        e2_left,
        debug.cross_e1_left_points,
        debug.cross_e2_left_points,
        "B: E1-L <-> E2-L cross camera",
    )
    pnp = _draw_matches(
        e1_left,
        e2_left,
        debug.surviving_e1_left_points,
        debug.surviving_e2_left_points,
        "C: valid triangulated shared tracks",
        inlier_mask=debug.surviving_inlier_mask,
    )
    composite = np.vstack([stereo, cross, pnp])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), composite):
        raise RuntimeError(f"Could not write debug image: {output_path}")

