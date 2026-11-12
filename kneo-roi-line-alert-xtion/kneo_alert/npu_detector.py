"""Person (+ optional face "is it a real person" check) detection on the Kneron KL730 NPU.

Both models live in ONE NEF (npu/compile_nef.py): YOLOv8n cut before DFL/Sigmoid (person)
and UltraFace RFB-320 cut before Softmax/decoding (face). The NPU runs the convolutions;
decoding + NMS run on the CPU (yolo_post.py / face_post.py).

detector.type:
  kneron_nef  KL730 NPU via Kneron PLUS (`kp`, preinstalled on the KNEO Pi image)
  cut_onnx    the same cut models on the CPU via OpenCV DNN, for testing on a PC

With detector.require_face: true, a person box only counts if a face is found in the
upper half of the box. This rejects person-shaped false positives without a face (coats,
chairs, cut-outs seen from behind). It does NOT reject photos of faces, and people facing
away from the camera are not counted.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np

from .config import DetectorConfig
from .detector import Detection, PersonDetector
from .face_post import INPUT_H as FACE_H, INPUT_W as FACE_W, decode_faces
from .yolo_post import decode_person

log = logging.getLogger(__name__)

YOLO_SIZE = 320
# KNEO Pi KL730 firmware 2.0.0 runs multi-model NEFs from toolchain <= v0.26 (the official
# examples use v0.25/v0.26). With v0.33.1 the SECOND model in a NEF times out on the NPU and can
# hang/reboot the board, so we refuse to run it.
MAX_MULTI_MODEL_TOOLCHAIN = (0, 26)
# Head output shapes (C, H, W) -> position in the list the decoders expect.
YOLO_ORDER = [(64, 40, 40), (80, 40, 40), (64, 20, 20), (80, 20, 20), (64, 10, 10), (80, 10, 10)]
FACE_ORDER = [(6, 30, 40), (12, 30, 40), (4, 15, 20), (8, 15, 20), (4, 8, 10), (8, 8, 10),
              (6, 4, 5), (12, 4, 5)]


def nef_toolchain_version(path: str) -> tuple[int, int] | None:
    """Toolchain version recorded inside a NEF ('kneron/toolchain:vX.Y...'), if present."""
    m = re.search(rb"kneron/toolchain:v(\d+)\.(\d+)", Path(path).read_bytes())
    return (int(m.group(1)), int(m.group(2))) if m else None


def _order_by_shape(outputs: list[np.ndarray], order: list[tuple[int, int, int]]) -> list[np.ndarray]:
    """NEF output node order isn't guaranteed to match ONNX order; head shapes are unique."""
    by_shape = {tuple(np.asarray(o).shape[-3:]): np.asarray(o).reshape(np.asarray(o).shape[-3:])
                for o in outputs}
    missing = [s for s in order if s not in by_shape]
    if missing:
        raise RuntimeError(f"Unexpected model outputs {sorted(by_shape)}; missing {missing}")
    return [by_shape[s] for s in order]


# ---------------------------------------------------------------------------- backends
class _NpuInputConverter:
    """Vectorized version of kneopi-examples' convert_onnx_data_to_npu_data (8/16-bit layouts).

    The scatter offsets depend only on the tensor shape, so they are computed once; each frame
    is then one quantize + one fancy-index assignment instead of a Python loop per element.
    """

    def __init__(self, kp, tensor_descriptor):
        L = kp.ModelTensorDataLayout
        layout = tensor_descriptor.data_layout
        if layout in (L.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B, L.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B,
                      L.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B, L.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B):
            self.dtype, self.mask16 = np.int8, False
        elif layout in (L.KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B, L.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B):
            self.dtype, self.mask16 = np.int16, True
        else:
            raise NotImplementedError(f"NPU input layout {layout} not supported by this converter")

        info = tensor_descriptor.tensor_shape_info
        shape = np.array(info.v2.shape, dtype=np.int64)
        stride_onnx = np.array(info.v2.stride_onnx, dtype=np.int64)
        stride_npu = np.array(info.v2.stride_npu, dtype=np.int64)
        idx = np.indices(tuple(shape)).reshape(len(shape), -1).T
        self.onnx_off = idx @ stride_onnx
        self.npu_off = idx @ stride_npu
        self.npu_len = int(max(shape * stride_npu))
        self.shape = tuple(int(s) for s in shape)

        q = tensor_descriptor.quantization_parameters.v1
        radix = np.array([d.radix for d in q.quantized_fixed_point_descriptor_list], dtype=np.float64)
        scale = np.array([d.scale.value for d in q.quantized_fixed_point_descriptor_list], dtype=np.float64)
        factor = np.power(2.0, radix) * scale
        if factor.size > 1:
            axes = tuple(d for d in range(len(shape)) if d != q.quantized_axis)
            factor = np.broadcast_to(np.expand_dims(factor, axis=axes), self.shape)
        self.factor = factor
        info_ = np.iinfo(self.dtype)
        self.lo, self.hi = info_.min, info_.max

    def __call__(self, x: np.ndarray) -> bytes:
        q = np.clip(np.round(x.reshape(self.shape) * self.factor), self.lo, self.hi).astype(self.dtype)
        if self.mask16:
            q = (q.view(np.uint16) & 0xFFFE).view(np.int16)
        buf = np.zeros(self.npu_len, dtype=self.dtype)
        buf[self.npu_off] = q.ravel()[self.onnx_off]
        return buf.tobytes()


class KneronNpuBackend:
    """One KL730 connection with a multi-model NEF; run(model_id, nchw) -> list of float outputs."""

    def __init__(self, nef_path: str, firmware_path: str, usb_port_id: int | None = None):
        import kp  # Kneron PLUS, preinstalled on the KNEO Pi image

        self.kp = kp
        for p in (nef_path, firmware_path):
            if not Path(p).is_file():
                raise FileNotFoundError(f"{p} not found (see README section 'NPU / NEF')")
        if usb_port_id is None:
            devices = kp.core.scan_devices()
            kl730 = [d for d in devices.device_descriptor_list
                     if d.product_id == kp.ProductId.KP_DEVICE_KL730.value]
            if not kl730:
                raise RuntimeError("No KL730 found (kp.core.scan_devices)")
            usb_port_id = kl730[0].usb_port_id
        self.dg = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        kp.core.set_timeout(device_group=self.dg, milliseconds=5000)
        kp.core.load_firmware_from_file(device_group=self.dg, scpu_fw_path=firmware_path, ncpu_fw_path="")
        nef = kp.core.load_model_from_file(device_group=self.dg, file_path=nef_path)
        self.models = {m.id: m for m in nef.models}
        self.convert = {mid: _NpuInputConverter(kp, m.input_nodes[0]) for mid, m in self.models.items()}
        self.toolchain = nef_toolchain_version(nef_path)
        # Models after the first one are only safe on this firmware with an older toolchain.
        self.unsafe_ids: set[int] = set()
        if len(nef.models) > 1 and self.toolchain and self.toolchain > MAX_MULTI_MODEL_TOOLCHAIN:
            self.unsafe_ids = {m.id for m in nef.models[1:]}
            log.error("NEF built with toolchain v%d.%d: model(s) %s would hang the KL730 (firmware supports "
                      "multi-model NEFs up to v%d.%d). Recompile with kneron/toolchain:v0.25.0.",
                      *self.toolchain, sorted(self.unsafe_ids), *MAX_MULTI_MODEL_TOOLCHAIN)
        log.info("KL730 ready on USB port %s; NEF models %s (toolchain %s)", usb_port_id, sorted(self.models),
                 "v%d.%d" % self.toolchain if self.toolchain else "?")

    def run(self, model_id: int, nchw: np.ndarray) -> list[np.ndarray]:
        kp = self.kp
        if model_id not in self.models:
            raise KeyError(f"model id {model_id} not in NEF (has {sorted(self.models)})")
        if model_id in self.unsafe_ids:
            raise RuntimeError(f"model {model_id} disabled: NEF toolchain too new for this firmware")
        desc = kp.GenericDataInferenceDescriptor(
            model_id=model_id, inference_number=0,
            input_node_data_list=[kp.GenericInputNodeData(buffer=self.convert[model_id](nchw))])
        kp.inference.generic_data_inference_send(device_group=self.dg, generic_inference_input_descriptor=desc)
        raw = kp.inference.generic_data_inference_receive(device_group=self.dg)
        return [kp.inference.generic_inference_retrieve_float_node(
                    node_idx=i, generic_raw_result=raw,
                    channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT).ndarray
                for i in range(raw.header.num_output_node)]

    def close(self) -> None:
        try:
            self.kp.core.disconnect_devices(device_group=self.dg)
        except Exception:  # noqa: BLE001
            pass


class CutOnnxBackend:
    """Same cut models on the CPU (OpenCV DNN) — for developing/testing on a PC."""

    def __init__(self, person_onnx: str, face_onnx: str | None):
        self.nets = {"person": cv2.dnn.readNetFromONNX(person_onnx)}
        if face_onnx:
            self.nets["face"] = cv2.dnn.readNetFromONNX(face_onnx)

    def run(self, which: str, nchw: np.ndarray) -> list[np.ndarray]:
        net = self.nets[which]
        net.setInput(nchw.astype(np.float32))
        return list(net.forward(net.getUnconnectedOutLayersNames()))


# ---------------------------------------------------------------------------- detector
class TwoModelDetector(PersonDetector):
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.last_faces: list[Detection] = []
        self.face_enabled = cfg.require_face
        if cfg.type == "kneron_nef":
            self.backend = KneronNpuBackend(cfg.nef_path, cfg.firmware_path)
            if self.face_enabled and cfg.face_model_id in self.backend.unsafe_ids:
                log.error("require_face disabled: face model can't run with this NEF; counting persons only")
                self.face_enabled = False
            self._run = lambda which, x: self.backend.run(
                cfg.person_model_id if which == "person" else cfg.face_model_id, x)
        else:  # cut_onnx
            self.backend = CutOnnxBackend(cfg.person_onnx, cfg.face_onnx if cfg.require_face else None)
            self._run = self.backend.run

    # --- person (YOLOv8n, 320x320 letterbox, RGB / 255) ---
    @staticmethod
    def _letterbox(frame: np.ndarray, size: int = YOLO_SIZE):
        h, w = frame.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
        return canvas, r, left, top

    def detect_persons(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img, r, px, py = self._letterbox(frame)
        x = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        heads = _order_by_shape(self._run("person", x), YOLO_ORDER)
        boxes, scores = decode_person([(heads[2 * i], heads[2 * i + 1]) for i in range(3)],
                                      conf_threshold=self.cfg.conf_threshold)
        return self._nms(boxes, scores, r, r, px, py, w, h, self.cfg.conf_threshold, self.cfg.nms_threshold)

    # --- face (UltraFace, 320x240 stretch, RGB, (x-127)/128) ---
    def detect_faces(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        img = cv2.cvtColor(cv2.resize(frame, (FACE_W, FACE_H)), cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((img - 127.0) / 128.0).transpose(2, 0, 1)[None]
        heads = _order_by_shape(self._run("face", x), FACE_ORDER)
        boxes, scores = decode_faces([(heads[2 * k], heads[2 * k + 1]) for k in range(4)],
                                     conf_threshold=self.cfg.face_conf)
        return self._nms(boxes, scores, FACE_W / w, FACE_H / h, 0, 0, w, h, self.cfg.face_conf, 0.3)

    @staticmethod
    def _nms(boxes, scores, rx, ry, px, py, w, h, conf, iou) -> list[Detection]:
        if len(boxes) == 0:
            return []
        xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
        keep = cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, iou)
        out = []
        for i in np.ravel(keep):
            x1, y1, x2, y2 = boxes[i]
            out.append(Detection(int(max(0, (x1 - px) / rx)), int(max(0, (y1 - py) / ry)),
                                 int(min(w - 1, (x2 - px) / rx)), int(min(h - 1, (y2 - py) / ry)),
                                 float(scores[i])))
        return out

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """All persons are returned (so turning away doesn't look like leaving); faces found in the
        frame are left in self.last_faces. The app uses them to require at least one face per visit."""
        persons = self.detect_persons(frame)
        if not self.face_enabled:
            self.last_faces = []
            return persons
        try:
            self.last_faces = self.detect_faces(frame) if persons else []
        except Exception as e:  # noqa: BLE001 - never let the face check take the app down
            log.error("Face detection failed (%s); require_face disabled, counting persons only", e)
            self.face_enabled = False
            self.last_faces = []
        return persons


def has_face(person: Detection, faces: list[Detection]) -> bool:
    """A face whose centre lies inside the upper half of the person box."""
    y_mid = person.y1 + (person.y2 - person.y1) / 2
    for f in faces:
        cx, cy = (f.x1 + f.x2) / 2, (f.y1 + f.y2) / 2
        if person.x1 <= cx <= person.x2 and person.y1 <= cy <= y_mid:
            return True
    return False
