"""Grab a few color frames from an Xtion / PrimeSense via OpenNI2 and save one as JPEG.

Usage (on the board):  .venv/bin/python deploy/check_openni2.py [/opt/OpenNI2]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from kneo_alert.camera import OpenNI2Capture  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
lib_dir = sys.argv[1] if len(sys.argv) > 1 else "/opt/OpenNI2"

from kneo_alert._openni_compat import import_openni2  # noqa: E402

openni2 = import_openni2()
openni2.initialize(lib_dir)
devices = openni2.Device.enumerate_uris() if hasattr(openni2.Device, "enumerate_uris") else []
print("OpenNI2 devices:", devices)
openni2.unload()

cap = OpenNI2Capture(lib_dir, 640, 480)
t0 = time.time()
ok_count, frame = 0, None
for _ in range(30):
    ok, f = cap.read()
    if ok:
        ok_count, frame = ok_count + 1, f
cap.release()
dt = time.time() - t0

print(f"Got {ok_count}/30 frames in {dt:.1f}s ({ok_count / dt:.1f} fps)")
if frame is not None:
    out = ROOT / "camera_test.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Frame {frame.shape[1]}x{frame.shape[0]} saved to {out}")
else:
    sys.exit("No frames received")
