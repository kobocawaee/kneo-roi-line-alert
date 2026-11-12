"""ROI polygon tests."""

from __future__ import annotations

import cv2
import numpy as np

from .config import RoiConfig
from .detector import Detection


class Roi:
    def __init__(self, cfg: RoiConfig, frame_size: tuple[int, int] | None = None):
        """frame_size=(w, h). Empty polygon means the whole frame."""
        self.cfg = cfg
        if cfg.polygon:
            self.poly = np.array(cfg.polygon, dtype=np.int32)
        elif frame_size:
            w, h = frame_size
            self.poly = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
        else:
            raise ValueError("ROI polygon is empty and frame size unknown")

    def contains_point(self, pt: tuple[int, int]) -> bool:
        return cv2.pointPolygonTest(self.poly.reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), False) >= 0

    def overlap_ratio(self, det: Detection) -> float:
        """Fraction of the bbox area that lies inside the polygon."""
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        mask = np.zeros((bh, bw), dtype=np.uint8)
        shifted = (self.poly - np.array([x1, y1])).astype(np.int32)
        cv2.fillPoly(mask, [shifted], 1)
        return float(mask.sum()) / (bw * bh)

    def is_inside(self, det: Detection) -> bool:
        if self.cfg.mode == "overlap":
            return self.overlap_ratio(det) >= self.cfg.min_overlap
        return self.contains_point(det.foot_point)

    def filter(self, dets: list[Detection]) -> list[Detection]:
        return [d for d in dets if self.is_inside(d)]

    def draw(self, frame: np.ndarray, color=(0, 200, 255)) -> None:
        cv2.polylines(frame, [self.poly.reshape(-1, 1, 2)], True, color, 2)
