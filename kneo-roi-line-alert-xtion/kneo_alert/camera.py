"""Frame sources with a cv2.VideoCapture-like interface: read() -> (ok, bgr_frame), release().

camera.source:
  0, 1, ...            local V4L2 / webcam index
  "video.mp4"          video file
  "rtsp://..."         network stream
  "openni2"            ASUS Xtion / PrimeSense via OpenNI2 (color stream)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

OPENNI2_SOURCE = "openni2"


@dataclass
class CameraConfig:
    source: Any = 0  # int index, video file path, RTSP/HTTP URL, or "openni2"
    openni2_path: str = "/opt/OpenNI2"  # dir containing libOpenNI2.so (for source: openni2)
    width: int | None = None
    height: int | None = None
    process_every_n_frames: int = 1
    reconnect_delay_sec: float = 3.0


class OpenNI2Capture:
    """Color stream of an OpenNI2 device (Xtion PRO Live etc.)."""

    def __init__(self, lib_dir: str, width: int | None = None, height: int | None = None,
                 timeout_sec: float = 2.0):
        from ._openni_compat import import_openni2  # pip install openni ; needs libOpenNI2.so
        openni2 = import_openni2()
        from openni import _openni2 as c_api

        self._openni2 = openni2
        self.timeout_sec = timeout_sec
        self.stream = None
        self.dev = None

        openni2.initialize(lib_dir)
        self.dev = openni2.Device.open_any()
        self.stream = self.dev.create_color_stream()

        # ctypes enum instances compare by identity, so compare their integer values.
        rgb = _enum_int(c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888)
        all_modes = self.stream.get_sensor_info().videoModes
        modes = [m for m in all_modes if _enum_int(m.pixelFormat) == rgb]
        if not modes:
            log.warning("No RGB888 color mode; available: %s",
                        [(_enum_int(m.pixelFormat), m.resolutionX, m.resolutionY, m.fps) for m in all_modes])
        log.info("OpenNI2 color modes: %s",
                 sorted({(m.resolutionX, m.resolutionY, m.fps) for m in modes}))
        if modes:
            want_w, want_h = width or 640, height or 480
            best = min(modes, key=lambda m: (abs(m.resolutionX - want_w) + abs(m.resolutionY - want_h),
                                             -m.fps))
            self.stream.set_video_mode(best)
            log.info("OpenNI2 color mode set to %dx%d@%d", best.resolutionX, best.resolutionY, best.fps)
        try:
            self.stream.set_mirroring_enabled(False)
        except Exception as e:  # not all devices support it
            log.debug("set_mirroring_enabled failed: %s", e)
        self.stream.start()

    def isOpened(self) -> bool:
        return self.stream is not None

    def read(self):
        if self.stream is None:
            return False, None
        ready = self._openni2.wait_for_any_stream([self.stream], timeout=self.timeout_sec)
        if ready is None:
            log.warning("OpenNI2: no frame within %.1fs", self.timeout_sec)
            return False, None
        frame = self.stream.read_frame()
        return True, openni_rgb_to_bgr(frame.get_buffer_as_uint8(), frame.width, frame.height,
                                       frame.stride)

    def set(self, *_args) -> bool:  # cv2.VideoCapture compatibility; resolution set in __init__
        return False

    def release(self) -> None:
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
            if self.dev is not None:
                self.dev.close()
        except Exception as e:
            log.debug("OpenNI2 release error: %s", e)
        finally:
            self.stream = None
            self.dev = None
            try:
                self._openni2.unload()
            except Exception:
                pass


def _enum_int(v) -> int:
    return int(v.value) if hasattr(v, "value") else int(v)


def openni_rgb_to_bgr(buf, width: int, height: int, stride: int | None = None) -> np.ndarray:
    """Convert an OpenNI2 RGB888 buffer (rows may be padded to `stride` bytes) into a BGR image copy."""
    arr = np.frombuffer(buf, dtype=np.uint8)
    stride = stride or width * 3
    img = arr[: stride * height].reshape(height, stride)[:, : width * 3].reshape(height, width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # also copies, so the OpenNI frame can be released


class _FailedCapture:
    """Stands in for a camera that couldn't be opened, so the main loop's reconnect logic handles it."""

    def isOpened(self) -> bool:
        return False

    def read(self):
        return False, None

    def release(self) -> None:
        pass


def open_camera(cfg: CameraConfig):
    if cfg.source == OPENNI2_SOURCE:
        try:
            return OpenNI2Capture(cfg.openni2_path, cfg.width, cfg.height)
        except ImportError:
            raise RuntimeError("camera.source is 'openni2' but the 'openni' package is missing: "
                               "pip install openni")
        except Exception as e:
            log.error("OpenNI2 open failed: %s", e)
            return _FailedCapture()

    cap = cv2.VideoCapture(cfg.source)
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    return cap
