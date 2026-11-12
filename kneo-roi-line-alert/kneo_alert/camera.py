"""Frame source (webcam / video file / network stream) via cv2.VideoCapture.

camera.source:
  0, 1, ...            local USB webcam index (/dev/video0, /dev/video1, ...)
  "video.mp4"          video file
  "rtsp://..."         network stream
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2


@dataclass
class CameraConfig:
    source: Any = 0  # int index, video file path, or RTSP/HTTP URL
    width: int | None = None
    height: int | None = None
    process_every_n_frames: int = 1
    reconnect_delay_sec: float = 3.0


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    if isinstance(cfg.source, int):
        # Prefer V4L2 on Linux for local cameras; fall back to OpenCV's default backend.
        cap = cv2.VideoCapture(cfg.source, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(cfg.source)
    else:
        cap = cv2.VideoCapture(cfg.source)
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    return cap
