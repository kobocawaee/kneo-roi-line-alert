"""Cut a YOLOv8 ONNX before its post-processing so it can be compiled for the Kneron NPU.

The NPU (and int8 quantization) handles convolutions well but not the DFL Softmax,
the class Sigmoid and the box decoding at the end of YOLOv8. We keep everything up to
the six final head convolutions and do the rest on the CPU (kneo_alert/yolo_post.py):

  outputs (stride 8, 16, 32):  box_s  (1, 64, H, W)  raw DFL logits (4 sides x 16 bins)
                               cls_s  (1, 80, H, W)  raw class logits (before sigmoid)

Usage (on the PC, needs `pip install onnx`):
    python npu/cut_yolov8.py models/yolov8n_320.onnx npu/yolov8n_320_cut.onnx
"""

from __future__ import annotations

import sys

import onnx
from onnx import utils as onnx_utils

STRIDES = (8, 16, 32)


def head_outputs() -> list[str]:
    outs = []
    for i in range(3):
        outs.append(f"/model.22/cv2.{i}/cv2.{i}.2/Conv_output_0")  # box (DFL logits)
        outs.append(f"/model.22/cv3.{i}/cv3.{i}.2/Conv_output_0")  # class logits
    return outs


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    names = {o for n in model.graph.node for o in n.output}
    missing = [o for o in head_outputs() if o not in names]
    if missing:
        sys.exit(f"Not a YOLOv8 export with the expected head names; missing: {missing}")
    onnx_utils.extract_model(src, dst, ["images"], head_outputs())
    cut = onnx.load(dst)
    onnx.checker.check_model(cut)
    ops = sorted({n.op_type for n in cut.graph.node})
    print(f"Saved {dst}")
    print("ops:", ops)
    print("outputs:", [o.name for o in cut.graph.output])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
