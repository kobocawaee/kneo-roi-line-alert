"""Xtion / OpenNI2 camera tests (Xtion version only)."""

import numpy as np


# ---------- Camera ----------

def test_openni_rgb_to_bgr_handles_stride():
    from kneo_alert.camera import openni_rgb_to_bgr
    w, h, stride = 4, 2, 16  # 12 bytes of pixels + 4 bytes padding per row
    buf = np.zeros(stride * h, np.uint8)
    buf[0:3] = [255, 0, 0]           # pixel (0,0) pure red in RGB
    buf[stride + 9:stride + 12] = [0, 0, 255]  # pixel (1,3) pure blue
    img = openni_rgb_to_bgr(buf.tobytes(), w, h, stride)
    assert img.shape == (2, 4, 3)
    assert img[0, 0].tolist() == [0, 0, 255]   # BGR red
    assert img[1, 3].tolist() == [255, 0, 0]   # BGR blue


def test_openni2_open_failure_falls_back_to_reconnect(monkeypatch):
    from kneo_alert import camera
    from kneo_alert.config import CameraConfig

    def boom(*a, **k):
        raise RuntimeError("no device")
    monkeypatch.setattr(camera, "OpenNI2Capture", boom)
    cap = camera.open_camera(CameraConfig(source="openni2"))
    assert cap.read() == (False, None)
    cap.release()
