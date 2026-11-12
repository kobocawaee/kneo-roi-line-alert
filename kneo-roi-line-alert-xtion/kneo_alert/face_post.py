"""CPU post-processing for UltraFace RFB-320 cut before Softmax/decoding (see npu/cut_ultraface.py).

heads: [(cls (A*2, H, W), reg (A*4, H, W)) for the 4 scales], raw NCHW head-conv outputs.
Returns face boxes in pixels of the 320x240 model input, and scores.
"""

from __future__ import annotations

import numpy as np

INPUT_W, INPUT_H = 320, 240
MIN_BOXES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STRIDES = [8, 16, 32, 64]
CENTER_VARIANCE, SIZE_VARIANCE = 0.1, 0.2


def _priors() -> list[np.ndarray]:
    """Per scale: (H*W*A, 4) priors as normalized (cx, cy, w, h), same order as the original model."""
    out = []
    for min_boxes, stride in zip(MIN_BOXES, STRIDES):
        fw, fh = int(np.ceil(INPUT_W / stride)), int(np.ceil(INPUT_H / stride))
        # The released model's priors use the (rounded-up) feature-map size here, e.g. 8 rows
        # for 240/32 = 7.5 -- verified against the original ONNX outputs.
        scale_w, scale_h = fw, fh
        js, is_ = np.meshgrid(np.arange(fh), np.arange(fw), indexing="ij")
        cx = ((is_ + 0.5) / scale_w).reshape(-1, 1)
        cy = ((js + 0.5) / scale_h).reshape(-1, 1)
        wh = np.array([[m / INPUT_W, m / INPUT_H] for m in min_boxes], np.float32)  # (A, 2)
        a = len(min_boxes)
        p = np.concatenate([np.repeat(cx, a, 1)[..., None], np.repeat(cy, a, 1)[..., None],
                            np.broadcast_to(wh, (cx.shape[0], a, 2))], axis=2).reshape(-1, 4)
        out.append(np.clip(p, 0.0, 1.0).astype(np.float32))
    return out


PRIORS = _priors()


def decode_faces(heads, conf_threshold: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    boxes_all, scores_all = [], []
    for (cls, reg), priors in zip(heads, PRIORS):
        cls = np.asarray(cls, np.float32)
        reg = np.asarray(reg, np.float32)
        # NCHW (A*k, H, W) -> (H, W, A*k) -> (H*W*A, k): same as the model's Transpose + Reshape.
        logits = cls.transpose(1, 2, 0).reshape(-1, 2)
        loc = reg.transpose(1, 2, 0).reshape(-1, 4)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        face = e[:, 1] / e.sum(axis=1)
        keep = face >= conf_threshold
        if not keep.any():
            continue
        p, l = priors[keep], loc[keep]
        cx = l[:, 0] * CENTER_VARIANCE * p[:, 2] + p[:, 0]
        cy = l[:, 1] * CENTER_VARIANCE * p[:, 3] + p[:, 1]
        w = np.exp(l[:, 2] * SIZE_VARIANCE) * p[:, 2]
        h = np.exp(l[:, 3] * SIZE_VARIANCE) * p[:, 3]
        boxes_all.append(np.stack([(cx - w / 2) * INPUT_W, (cy - h / 2) * INPUT_H,
                                   (cx + w / 2) * INPUT_W, (cy + h / 2) * INPUT_H], axis=1))
        scores_all.append(face[keep])
    if not boxes_all:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
    return np.concatenate(boxes_all), np.concatenate(scores_all)


def decode_all(heads) -> tuple[np.ndarray, np.ndarray]:
    """Scores (N, 2) after softmax and boxes (N, 4) normalized corners: matches the uncut model outputs."""
    sc, bx = [], []
    for (cls, reg), priors in zip(heads, PRIORS):
        logits = np.asarray(cls, np.float32).transpose(1, 2, 0).reshape(-1, 2)
        loc = np.asarray(reg, np.float32).transpose(1, 2, 0).reshape(-1, 4)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        sc.append(e / e.sum(axis=1, keepdims=True))
        cx = loc[:, 0] * CENTER_VARIANCE * priors[:, 2] + priors[:, 0]
        cy = loc[:, 1] * CENTER_VARIANCE * priors[:, 3] + priors[:, 1]
        w = np.exp(loc[:, 2] * SIZE_VARIANCE) * priors[:, 2]
        h = np.exp(loc[:, 3] * SIZE_VARIANCE) * priors[:, 3]
        bx.append(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1))
    return np.concatenate(sc), np.concatenate(bx)
