"""Grab calibration frames from the real camera for NPU quantization.

Run on the board while people walk in/out, stand at different distances, lights on/off...
Variety matters more than count.

    .venv/bin/python npu/capture_calib.py --count 100 --interval 1.0
    # -> calib/calib_000.jpg ... (copy this folder to the PC for npu/compile_nef.py)

Note: these pictures contain people; they are git-ignored, keep them private.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kneo_alert.camera import open_camera  # noqa: E402
from kneo_alert.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between saved frames")
    ap.add_argument("--out", default=str(ROOT / "calib"))
    args = ap.parse_args()

    cfg = load_config(args.config, ROOT / ".env")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(cfg.camera)
    saved, last = 0, 0.0
    print(f"Saving {args.count} frames to {out}/ every {args.interval}s — move around in front of the camera. Ctrl+C to stop.")
    try:
        while saved < args.count:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            if now - last >= args.interval:
                cv2.imwrite(str(out / f"calib_{saved:03d}.jpg"), frame)
                saved += 1
                last = now
                print(f"\r{saved}/{args.count}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    print(f"\nDone: {saved} frames in {out}/")


if __name__ == "__main__":
    main()
