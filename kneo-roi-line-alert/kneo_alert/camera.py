"""Frame source (webcam / video file / network stream) via cv2.VideoCapture.

camera.source:
  auto                 first USB webcam (skips the KNEO Pi's own vpl_voc display outputs).
                       Recommended: /dev/videoN numbers change between reboots.
  /dev/v4l/by-id/...   a specific webcam by its stable udev name (also /dev/videoN paths)
  0, 1, ...            local webcam index (/dev/video0, /dev/video1, ...)
  "video.mp4"          video file
  "rtsp://..."         network stream
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

log = logging.getLogger(__name__)

SYS_V4L = Path("/sys/class/video4linux")
SKIP_NAMES = ("vpl_voc",)  # KNEO Pi video OUTPUT channels, not cameras


@dataclass
class CameraConfig:
    source: Any = "auto"  # "auto", /dev/v4l/by-id/... path, int index, video file, or RTSP/HTTP URL
    width: int | None = None
    height: int | None = None
    process_every_n_frames: int = 1
    reconnect_delay_sec: float = 3.0


def find_webcam_index(sys_dir: Path = SYS_V4L) -> int | None:
    """First capture node of a USB camera: /sys/class/video4linux/videoN with index 0
    (a UVC webcam's second node, index 1, is metadata) whose name isn't a known non-camera."""
    candidates = []
    for d in sys_dir.glob("video*"):
        m = re.fullmatch(r"video(\d+)", d.name)
        if not m:
            continue
        name = (d / "name").read_text(errors="replace").strip() if (d / "name").exists() else ""
        index = (d / "index").read_text().strip() if (d / "index").exists() else "0"
        if index != "0" or name.startswith(SKIP_NAMES):
            continue
        candidates.append((int(m.group(1)), name))
    if not candidates:
        return None
    num, name = min(candidates)
    log.info("camera.source auto -> /dev/video%d (%s)", num, name)
    return num


def resolve_source(source: Any) -> Any:
    """Turn 'auto' and /dev/... paths into a numeric index OpenCV's V4L2 backend can open."""
    if isinstance(source, str):
        if source.lower() == "auto":
            idx = find_webcam_index()
            if idx is None:
                raise RuntimeError("camera.source: auto found no USB camera in /sys/class/video4linux")
            return idx
        if source.startswith("/dev/"):
            real = os.path.realpath(source)
            m = re.fullmatch(r"/dev/video(\d+)", real)
            if not m:
                raise RuntimeError(f"camera.source {source!r} -> {real!r} is not a /dev/videoN device")
            return int(m.group(1))
    return source


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    source = resolve_source(cfg.source)
    if isinstance(source, int):
        # Prefer V4L2 on Linux for local cameras; fall back to OpenCV's default backend.
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else None
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    return cap
