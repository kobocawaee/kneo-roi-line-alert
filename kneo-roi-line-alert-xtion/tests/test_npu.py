"""NPU path tests that run without the KL730: input conversion, output ordering, face check."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from kneo_alert.detector import Detection
from kneo_alert.npu_detector import YOLO_ORDER, _NpuInputConverter, _order_by_shape, has_face


class FakeLayout:
    KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B = 1
    KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B = 2
    KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B = 3
    KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B = 4
    KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B = 5
    KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B = 6
    KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL = 7


FAKE_KP = NS(ModelTensorDataLayout=FakeLayout)


def make_descriptor(shape, stride_npu, radix=7, scale=1.0, layout=FakeLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B):
    stride_onnx = [int(np.prod(shape[i + 1:])) for i in range(len(shape))]
    return NS(
        data_layout=layout,
        tensor_shape_info=NS(v2=NS(shape=list(shape), stride_onnx=stride_onnx, stride_npu=list(stride_npu))),
        quantization_parameters=NS(v1=NS(quantized_axis=1, quantized_fixed_point_descriptor_list=[
            NS(radix=radix, scale=NS(value=scale))])),
    )


def reference_convert(desc, x):
    """Straight port of kneopi-examples utils/ExampleHelper.convert_onnx_data_to_npu_data (8-bit path)."""
    info = desc.tensor_shape_info.v2
    shape = np.array(info.shape)
    d = desc.quantization_parameters.v1.quantized_fixed_point_descriptor_list[0]
    q = np.clip(np.round(x * (2.0 ** d.radix * d.scale.value)), -128, 127).astype(np.int8).flatten()
    out = np.zeros(int(max(shape * np.array(info.stride_npu))), dtype=np.int8)
    idx = np.zeros(len(shape), dtype=int)
    while True:
        out[idx.dot(info.stride_npu)] = q[idx.dot(info.stride_onnx)]
        for dim in range(len(shape) - 1, -1, -1):
            idx[dim] += 1
            if idx[dim] == shape[dim]:
                if dim == 0:
                    break
                idx[dim] = 0
                continue
            break
        if idx[0] == shape[0]:
            break
    return out.tobytes()


def test_vectorized_converter_matches_official_loop():
    shape = (1, 3, 6, 8)
    # A padded, channel-interleaved NPU layout (like 4W4C8B): C padded to 4, W padded to 8.
    stride_npu = (4 * 6 * 8, 1, 8 * 4, 4)
    desc = make_descriptor(shape, stride_npu, radix=6, scale=1.25)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=shape).astype(np.float32)
    assert _NpuInputConverter(FAKE_KP, desc)(x) == reference_convert(desc, x)


def test_converter_rejects_unsupported_layout():
    desc = make_descriptor((1, 3, 2, 2), (12, 4, 2, 1), layout=FakeLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL)
    with pytest.raises(NotImplementedError):
        _NpuInputConverter(FAKE_KP, desc)


def test_outputs_reordered_by_shape():
    shuffled = [np.full((1,) + s, i, np.float32) for i, s in enumerate(YOLO_ORDER)][::-1]
    ordered = _order_by_shape(shuffled, YOLO_ORDER)
    assert [o.shape for o in ordered] == YOLO_ORDER
    assert [int(o[0, 0, 0]) for o in ordered] == list(range(len(YOLO_ORDER)))  # values travel with shapes


def test_has_face_requires_face_in_upper_half():
    person = Detection(100, 100, 200, 400, 0.9)
    assert has_face(person, [Detection(130, 110, 170, 160, 0.9)])       # head
    assert not has_face(person, [Detection(130, 300, 170, 340, 0.9)])   # lower half
    assert not has_face(person, [Detection(300, 110, 340, 160, 0.9)])   # outside the box
    assert not has_face(person, [])


def test_nef_toolchain_version_is_read_from_file(tmp_path):
    from kneo_alert.npu_detector import (KNOWN_BAD_MULTI_MODEL_TOOLCHAIN, LAST_TESTED_GOOD_TOOLCHAIN,
                                         nef_toolchain_version)
    new = tmp_path / "new.nef"
    new.write_bytes(b"\x00\x01junk kneron/toolchain:v0.33.1\x00more")
    old = tmp_path / "old.nef"
    old.write_bytes(b"xx kneron/toolchain:v0.25.0_3 yy")
    none = tmp_path / "none.nef"
    none.write_bytes(b"no version here")
    assert nef_toolchain_version(str(new)) == (0, 33) >= KNOWN_BAD_MULTI_MODEL_TOOLCHAIN
    assert nef_toolchain_version(str(old)) == (0, 25) <= LAST_TESTED_GOOD_TOOLCHAIN
    # v0.31 (reported working elsewhere) is not blocked, only warned about
    assert LAST_TESTED_GOOD_TOOLCHAIN < (0, 31) < KNOWN_BAD_MULTI_MODEL_TOOLCHAIN
    assert nef_toolchain_version(str(none)) is None


def test_face_failure_falls_back_to_persons_only():
    from kneo_alert.config import DetectorConfig
    from kneo_alert.npu_detector import TwoModelDetector

    det = TwoModelDetector.__new__(TwoModelDetector)
    det.cfg = DetectorConfig(type="cut_onnx", require_face=True)
    det.face_enabled = True
    det.last_faces = []
    person = Detection(10, 10, 100, 300, 0.9)
    det.detect_persons = lambda frame: [person]

    def boom(frame):
        raise RuntimeError("KP_FW_INFERENCE_TIMEOUT_103")
    det.detect_faces = boom
    frame = np.zeros((480, 640, 3), np.uint8)
    assert det.detect(frame) == [person]      # person still counted
    assert det.face_enabled is False           # face check switched off, no more NPU calls
    assert det.detect(frame) == [person]


def _run_visit(tmp_path, face_at, seconds=10, fps=10):
    """One person stands in the ROI the whole time; face_at(t) says if their face is visible."""
    import cv2
    from pathlib import Path

    from kneo_alert import app
    from kneo_alert.config import load_config

    vid = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    for _ in range(seconds * fps):
        w.write(np.full((480, 640, 3), 60, np.uint8))
    w.release()

    class FakeTwoModel:
        face_enabled = True
        last_faces: list = []
        i = 0

        def detect(self, frame):
            FakeTwoModel.i += 1
            t = FakeTwoModel.i / fps
            person = Detection(300, 100, 400, 450, 0.9) if 1 <= t < 8 else None
            FakeTwoModel.last_faces = [Detection(330, 110, 370, 160, 0.9)] if person and face_at(t) else []
            return [person] if person else []

    orig = app.create_detector
    app.create_detector = lambda _c: FakeTwoModel()
    try:
        cfg = load_config(Path(__file__).parent.parent / "config.example.yaml", "none.env")
        cfg.camera.source = str(vid)
        cfg.roi.polygon = [[250, 50], [600, 50], [600, 470], [250, 470]]
        cfg.roi.mode = "overlap"
        cfg.alert.save_snapshot = False
        cfg.path = None
        return app.run(cfg, dry_run=True)
    finally:
        app.create_detector = orig


def test_require_face_turning_around_is_still_one_alert(tmp_path, caplog):
    """Board 2026-11-12: face visible, then turned away for 2 s, then visible again -> was counted
    as 2 people because faceless frames dropped the person. Now: one person, one alert."""
    import logging
    caplog.set_level(logging.INFO)
    alerts = _run_visit(tmp_path, face_at=lambda t: t < 2.5 or t > 4.5)
    assert alerts == 1
    assert "ALERT: 1 person(s)" in caplog.text and "arrived" not in caplog.text


def test_require_face_back_to_camera_whole_visit_no_alert(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    assert _run_visit(tmp_path, face_at=lambda t: False) == 0
    assert "No face seen during this visit" in caplog.text


def test_faceless_visit_does_not_start_cooldown(tmp_path, caplog):
    """Board 2026-11-13: a back-to-camera visit was skipped, but its cooldown then blocked the
    real (face-visible) visit 12 s later."""
    import logging
    caplog.set_level(logging.INFO)
    # visit 1 (1-8 s) no face; ROI empty 8-? ; we reuse _run_visit with face only after 9 s won't
    # work (person leaves at 8 s), so drive the trigger + gate directly:
    from kneo_alert.config import TriggerConfig
    from kneo_alert.trigger import EntryTrigger

    clock = type("C", (), {"t": 0.0, "__call__": lambda self: self.t})()
    t = EntryTrigger(TriggerConfig(enter_frames=2, exit_sec=2, leave_sec=1, gather_sec=3, cooldown_sec=60), clock)
    fired = []
    for i in range(400):                      # 0.1 s steps
        clock.t = i * 0.1
        count = 1 if (1 <= clock.t < 6 or 12 <= clock.t < 20) else 0
        r = t.update(count)
        if r:
            if clock.t < 10:                  # first visit: nobody showed a face -> skipped
                t.cancel_cooldown()
            else:
                fired.append(round(clock.t))
    assert fired == [15]                      # second visit alerts despite being < 60 s later
