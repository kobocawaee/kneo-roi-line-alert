"""Click points on a camera frame to define the ROI polygon.

Usage:
    python tools/pick_roi.py                 # uses camera.source from config.yaml
    python tools/pick_roi.py --source 0      # or a file / RTSP URL / image path

Controls: left-click = add point, right-click / u = undo, c = clear,
          s or Enter = print & save to config.yaml, q / Esc = quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def grab_frame(source, cam: dict):
    if isinstance(source, str) and Path(source).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        img = cv2.imread(source)
        if img is None:
            sys.exit(f"Cannot read image {source}")
        return img
    sys.path.insert(0, str(ROOT))
    from kneo_alert.camera import CameraConfig, open_camera

    known = CameraConfig.__dataclass_fields__
    cfg = CameraConfig(**{k: v for k, v in cam.items() if k in known})
    cfg.source = source
    cap = open_camera(cfg)
    frame = None
    for _ in range(10):  # let auto-exposure settle
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        sys.exit(f"Cannot read a frame from {source!r}")
    return frame


def save_polygon(config_path: Path, points: list[list[int]]) -> None:
    sys.path.insert(0, str(ROOT))
    from kneo_alert.config import save_roi_polygon
    try:
        save_roi_polygon(config_path, points)
    except ValueError as e:
        sys.exit(f"{e}; paste the line manually.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--source")
    args = ap.parse_args()

    config_path = Path(args.config)
    cam = {}
    if config_path.is_file():
        cam = (yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}).get("camera", {}) or {}
    source = args.source if args.source is not None else cam.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    base = grab_frame(source, cam)
    h, w = base.shape[:2]
    print(f"Frame size: {w}x{h}  (ROI coords are in this resolution)")
    points: list[list[int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([int(x), int(y)])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    win = "pick ROI (s=save, u=undo, c=clear, q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = base.copy()
        if points:
            pts = np.array(points, np.int32).reshape(-1, 1, 2)
            if len(points) >= 3:
                overlay = img.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 255))
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
            cv2.polylines(img, [pts], len(points) >= 3, (0, 200, 255), 2)
            for p in points:
                cv2.circle(img, tuple(p), 4, (0, 0, 255), -1)
        cv2.imshow(win, img)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u") and points:
            points.pop()
        elif key == ord("c"):
            points.clear()
        elif key in (ord("s"), 13):
            if len(points) < 3:
                print("Need at least 3 points")
                continue
            print(f"roi.polygon: {points}")
            if config_path.is_file():
                save_polygon(config_path, points)
                print(f"Saved to {config_path}")
            cv2.imwrite(str(ROOT / "roi_preview.jpg"), img)
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
