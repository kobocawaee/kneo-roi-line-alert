"""Person detectors. All return a list of Detection in original-frame pixel coords.

To use a board-specific NPU (e.g. a vendor SDK on Kneo Pi), subclass
`PersonDetector`, implement `detect`, and register it in `create_detector`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig

COCO_PERSON_CLASS = 0


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def foot_point(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, self.y2)


class PersonDetector:
    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class HogPersonDetector(PersonDetector):
    """OpenCV built-in HOG+SVM people detector. No model file needed; less accurate."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        if not hasattr(cv2, "HOGDescriptor"):
            raise RuntimeError(
                f"OpenCV {cv2.__version__} has no HOGDescriptor (removed in 5.x). "
                "Install opencv-python<5, or use detector.type: yolo_onnx."
            )
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        scale = 1.0
        if w > 640:  # HOG is slow on large frames
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        boxes, scores = [], []
        for (x, y, bw, bh), s in zip(rects, np.ravel(weights) if len(weights) else []):
            boxes.append([int(x), int(y), int(bw), int(bh)])
            scores.append(float(s))
        # HOG weights are SVM margins, not probabilities; use a loose threshold.
        keep = cv2.dnn.NMSBoxes(boxes, scores, 0.3, self.cfg.nms_threshold) if boxes else []
        out = []
        for i in np.ravel(keep):
            x, y, bw, bh = boxes[i]
            out.append(Detection(
                int(x / scale), int(y / scale), int((x + bw) / scale), int((y + bh) / scale), scores[i],
            ))
        return out


class YoloOnnxDetector(PersonDetector):
    """YOLOv8/YOLO11 ONNX (COCO) run through OpenCV DNN. Output shape (1, 84, N)."""

    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        path = Path(cfg.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"ONNX model not found: {path}. See README section 'YOLO 模型' to export one, "
                "or set detector.type: hog."
            )
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.size = cfg.input_size

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = frame.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img, r, pad_x, pad_y = self._letterbox(frame)
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.size, self.size), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()
        return self.postprocess(out, r, pad_x, pad_y, w, h)

    def postprocess(self, out: np.ndarray, r: float, pad_x: int, pad_y: int, w: int, h: int) -> list[Detection]:
        preds = np.squeeze(out, axis=0)
        if preds.shape[0] == 4 + 80:  # (84, N) -> (N, 84)
            preds = preds.T
        # preds rows: cx, cy, w, h, 80 class scores
        class_scores = preds[:, 4:]
        cls = np.argmax(class_scores, axis=1)
        conf = class_scores[np.arange(len(cls)), cls]
        mask = (cls == COCO_PERSON_CLASS) & (conf >= self.cfg.conf_threshold)
        preds, conf = preds[mask], conf[mask]
        if len(preds) == 0:
            return []

        boxes = []
        for cx, cy, bw, bh in preds[:, :4]:
            x = (cx - bw / 2 - pad_x) / r
            y = (cy - bh / 2 - pad_y) / r
            boxes.append([int(x), int(y), int(bw / r), int(bh / r)])
        keep = cv2.dnn.NMSBoxes(boxes, conf.tolist(), self.cfg.conf_threshold, self.cfg.nms_threshold)
        out_dets = []
        for i in np.ravel(keep):
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
            out_dets.append(Detection(x1, y1, x2, y2, float(conf[i])))
        return out_dets


def create_detector(cfg: DetectorConfig) -> PersonDetector:
    if cfg.type == "hog":
        return HogPersonDetector(cfg)
    if cfg.type == "yolo_onnx":
        return YoloOnnxDetector(cfg)
    if cfg.type in ("kneron_nef", "cut_onnx"):
        from .npu_detector import TwoModelDetector
        return TwoModelDetector(cfg)
    raise ValueError(f"Unknown detector.type: {cfg.type!r} "
                     "(use 'kneron_nef', 'cut_onnx', 'yolo_onnx' or 'hog')")
