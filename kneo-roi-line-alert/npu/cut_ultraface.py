"""Cut UltraFace (version-RFB-320.onnx) before its Softmax/box decoding for the Kneron NPU.

Model: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB (MIT)
Input 1x3x240x320, RGB, (x - 127) / 128.

Kept outputs: the 8 head convolutions (4 scales x [class logits, box offsets]),
NCHW, before the Transpose/Reshape/Softmax/Exp tail:
  scale 0 (40x30, 3 anchors): cls 6ch, reg 12ch
  scale 1 (20x15, 2 anchors): cls 4ch, reg 8ch
  scale 2 (10x8,  2 anchors): cls 4ch, reg 8ch
  scale 3 (5x4,   3 anchors): cls 6ch, reg 12ch
The rest is done on the CPU by kneo_alert/face_post.py.

Usage:  python npu/cut_ultraface.py version-RFB-320.onnx npu/ultraface_320_cut.onnx
"""

from __future__ import annotations

import sys

import onnx
from onnx import utils as onnx_utils


def head_outputs(model: onnx.ModelProto) -> list[str]:
    """Inputs of the Transposes that start the head tail = the 8 head conv outputs, in graph order."""
    producers = {o: n for n in model.graph.node for o in n.output}
    outs = []
    for n in model.graph.node:
        if n.op_type == "Transpose":
            src = n.input[0]
            if producers.get(src) is not None and producers[src].op_type == "Conv":
                outs.append(src)
    return outs


def main(src: str, dst: str) -> None:
    model = onnx.load(src)
    outs = head_outputs(model)
    if len(outs) != 8:
        sys.exit(f"Expected 8 head convolutions, found {len(outs)}: is this version-RFB-320.onnx?")
    # Old exporter lists weights as graph inputs; only the image is a real input.
    init = {i.name for i in model.graph.initializer}
    real_inputs = [i.name for i in model.graph.input if i.name not in init]
    onnx_utils.extract_model(src, dst, real_inputs, outs)
    cut = onnx.load(dst)
    onnx.checker.check_model(cut)
    print(f"Saved {dst}")
    print("ops:", sorted({n.op_type for n in cut.graph.node}))
    print("outputs (cls0, reg0, cls1, reg1, cls2, reg2, cls3, reg3):", [o.name for o in cut.graph.output])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
