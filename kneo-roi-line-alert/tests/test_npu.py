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
