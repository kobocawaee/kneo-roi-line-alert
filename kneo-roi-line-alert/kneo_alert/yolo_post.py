"""CPU post-processing for a YOLOv8 head that was cut before DFL/Sigmoid (see npu/cut_yolov8.py).

Input: for each stride (8, 16, 32) the raw head maps
    box (64, H, W): 4 box sides x 16 DFL bins (logits)
    cls (80, H, W): class logits
Output: person boxes in model-input pixel coordinates (x1, y1, x2, y2, score).
"""

from __future__ import annotations

import numpy as np

REG_MAX = 16
PERSON = 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def decode_person(heads: list[tuple[np.ndarray, np.ndarray]], strides=(8, 16, 32),
                  conf_threshold: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """heads: [(box (64,H,W), cls (80,H,W)) per stride]. Returns (boxes Nx4 xyxy, scores N)."""
    all_boxes, all_scores = [], []
    bins = np.arange(REG_MAX, dtype=np.float32)
    for (box, cls), stride in zip(heads, strides):
        box = np.asarray(box, dtype=np.float32).reshape(4 * REG_MAX, -1)
        cls = np.asarray(cls, dtype=np.float32)
        h, w = cls.shape[-2:]
        # Only the person channel matters: skip everything else early (cheap on the board CPU).
        person = _sigmoid(cls.reshape(cls.shape[0], -1)[PERSON])
        keep = np.nonzero(person >= conf_threshold)[0]
        if keep.size == 0:
            continue
        dfl = _softmax(box[:, keep].reshape(4, REG_MAX, -1), axis=1)   # (4, 16, K)
        dist = (dfl * bins[None, :, None]).sum(axis=1)                    # (4, K): l, t, r, b
        ys, xs = np.divmod(keep, w)
        cx, cy = xs + 0.5, ys + 0.5
        x1 = (cx - dist[0]) * stride
        y1 = (cy - dist[1]) * stride
        x2 = (cx + dist[2]) * stride
        y2 = (cy + dist[3]) * stride
        all_boxes.append(np.stack([x1, y1, x2, y2], axis=1))
        all_scores.append(person[keep])
    if not all_boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(all_boxes), np.concatenate(all_scores)
