import numpy as np
import pytest
import requests

from kneo_alert.config import DetectorConfig, LineConfig, RoiConfig, TriggerConfig, load_config
from kneo_alert.detector import Detection, YoloOnnxDetector
from kneo_alert.notifier import PUSH_URL, LineMessagingNotifier, LineNotifyError
from kneo_alert.roi import Roi
from kneo_alert.trigger import EntryTrigger

# ---------- ROI ----------

SQUARE = [[100, 100], [300, 100], [300, 300], [100, 300]]


def test_foot_point_inside_and_outside():
    roi = Roi(RoiConfig(polygon=SQUARE, mode="foot"))
    assert roi.is_inside(Detection(150, 50, 250, 200, 0.9))      # foot (200,200) inside
    assert not roi.is_inside(Detection(150, 250, 250, 400, 0.9))  # foot (200,400) outside
    # Head inside but feet below the ROI: not counted in foot mode
    assert not roi.is_inside(Detection(150, 150, 250, 350, 0.9))


def test_overlap_mode():
    roi = Roi(RoiConfig(polygon=SQUARE, mode="overlap", min_overlap=0.3))
    assert roi.overlap_ratio(Detection(100, 100, 200, 200, 1)) == pytest.approx(1.0, abs=0.05)
    assert roi.overlap_ratio(Detection(0, 0, 50, 50, 1)) == 0.0
    # half the box inside
    assert roi.overlap_ratio(Detection(200, 100, 400, 200, 1)) == pytest.approx(0.5, abs=0.05)
    assert roi.is_inside(Detection(200, 100, 400, 200, 1))


def test_empty_polygon_means_whole_frame():
    roi = Roi(RoiConfig(polygon=[]), frame_size=(640, 480))
    assert roi.is_inside(Detection(10, 10, 50, 470, 1))


# ---------- Trigger ----------

class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def feed(trigger, seq):
    return [bool(trigger.update(c)) for c in seq]


def T(**kw):
    kw.setdefault("gather_sec", 0)
    return TriggerConfig(**kw)


def test_trigger_requires_consecutive_frames():
    t = EntryTrigger(T(enter_frames=3, exit_frames=2, cooldown_sec=0), FakeClock())
    assert feed(t, [1, 0, 1, 1]) == [False] * 4
    assert bool(t.update(1)) is True


def test_trigger_fires_once_while_occupied():
    t = EntryTrigger(T(enter_frames=2, exit_frames=3, cooldown_sec=0), FakeClock())
    fired = feed(t, [1] * 50)
    assert sum(fired) == 1
    # brief dropout shorter than exit_frames doesn't re-arm
    assert sum(feed(t, [0, 0, 1, 1, 1])) == 0


def test_trigger_rearms_after_exit():
    t = EntryTrigger(T(enter_frames=2, exit_frames=3, cooldown_sec=0), FakeClock())
    assert sum(feed(t, [1, 1, 0, 0, 0, 1, 1])) == 2


def test_trigger_cooldown():
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=1, exit_frames=1, cooldown_sec=60), clock)
    assert t.update(1)
    t.update(0)
    clock.t = 30
    assert not t.update(1)  # in cooldown
    t.update(0)
    clock.t = 61
    assert t.update(1)


def test_gather_reports_people_arriving_seconds_apart():
    """User's case: I walk in, my friend follows 2 s later -> one alert saying 2 people."""
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=2, exit_frames=5, cooldown_sec=60, gather_sec=5), clock)
    results = []
    for i, count in enumerate([1] * 8 + [2] * 30):  # 4 detections/s; friend appears at 2 s
        clock.t = i * 0.25
        results.append(t.update(count))
    fired = [r for r in results if r]
    assert len(fired) == 1
    assert fired[0].count == 2
    # alert fires gather_sec after the first person was confirmed (2nd frame, t=0.25)
    assert results.index(fired[0]) == int((0.25 + 5) / 0.25)


def test_gather_ignores_single_frame_spike():
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=3, exit_frames=5, gather_sec=2), clock)
    seq = [1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1]  # one bogus frame with 3 boxes
    fired = []
    for i, c in enumerate(seq):
        clock.t = i * 0.25
        r = t.update(c)
        if r:
            fired.append(r.count)
    assert fired == [1]


def run_seq(t, clock, seq, dt=0.25):
    out = []
    for i, c in enumerate(seq):
        clock.t = i * dt
        out.append(t.update(c))
    return out


def test_door_one_after_another_counts_both():
    """Narrow door: I walk through the ROI and leave it, then my friend walks in."""
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=2, exit_frames=15, gather_sec=5), clock)
    seq = [1] * 6 + [0] * 3 + [1] * 6 + [0] * 10  # me (1.5 s), gap (0.75 s), friend
    res = run_seq(t, clock, seq)
    fired = [r for r in res if r]
    assert len(fired) == 1 and fired[0].count == 2
    events = [(r.arrived, r.same_group) for r in res if r.arrived]
    assert events == [(1, False), (1, False)]  # two separate arrival events -> two pictures


def test_group_then_latecomer():
    """Two walk in together, a third comes later after they left the ROI."""
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=2, exit_frames=15, gather_sec=5), clock)
    seq = [2] * 6 + [0] * 3 + [1] * 6 + [0] * 10
    res = run_seq(t, clock, seq)
    events = [r.arrived for r in res if r.arrived]
    assert events == [2, 1]
    assert [r.count for r in res if r] == [3]


def test_joining_people_still_inside_is_same_group():
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=2, exit_frames=15, gather_sec=5), clock)
    res = run_seq(t, clock, [1] * 4 + [2] * 8)
    events = [(r.arrived, r.same_group) for r in res if r.arrived]
    assert events == [(1, False), (1, True)]  # second joined while first still there -> one picture


def test_single_missed_detection_is_not_a_new_arrival():
    clock = FakeClock()
    t = EntryTrigger(T(enter_frames=2, exit_frames=15, leave_frames=2, gather_sec=5), clock)
    res = run_seq(t, clock, [1, 1, 1, 0, 1, 1, 1, 1] + [1] * 20)
    assert [r.count for r in res if r] == [1]


# ---------- YOLO postprocess ----------

def test_yolo_postprocess_maps_back_to_frame():
    det = YoloOnnxDetector.__new__(YoloOnnxDetector)
    det.cfg = DetectorConfig(conf_threshold=0.5, nms_threshold=0.45)
    out = np.zeros((1, 84, 3), dtype=np.float32)
    # anchor 0: person, conf 0.9, box centred (320,320) 100x200 in 640 letterbox space
    out[0, :4, 0] = [320, 320, 100, 200]
    out[0, 4, 0] = 0.9
    # anchor 1: car (class 2), should be dropped
    out[0, :4, 1] = [100, 100, 50, 50]
    out[0, 4 + 2, 1] = 0.95
    # anchor 2: person but low confidence
    out[0, :4, 2] = [200, 200, 50, 50]
    out[0, 4, 2] = 0.2
    # 640x480 frame letterboxed into 640x640: r=1, pad_y=80
    dets = det.postprocess(out, r=1.0, pad_x=0, pad_y=80, w=640, h=480)
    assert len(dets) == 1
    d = dets[0]
    assert (d.x1, d.y1, d.x2, d.y2) == (270, 140, 370, 340)
    assert d.score == pytest.approx(0.9)


# ---------- Notifier ----------

class FakeResp:
    def __init__(self, status, text=""):
        self.status_code = status
        self.text = text


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append((url, json, headers))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def make_notifier(session, to=("Uabc",)):
    cfg = LineConfig(channel_access_token="tok", to=list(to))
    return LineMessagingNotifier(cfg, session=session, max_retries=3)


def test_push_success_payload(monkeypatch):
    s = FakeSession([FakeResp(200)])
    make_notifier(s).push_text("hi")
    url, body, headers = s.calls[0]
    assert url == PUSH_URL
    assert body == {"to": "Uabc", "messages": [{"type": "text", "text": "hi"}]}
    assert headers["Authorization"] == "Bearer tok"
    assert headers["X-Line-Retry-Key"]


def test_push_retries_5xx_with_same_retry_key(monkeypatch):
    monkeypatch.setattr("kneo_alert.notifier.time.sleep", lambda _: None)
    s = FakeSession([FakeResp(500), requests.ConnectionError("x"), FakeResp(200)])
    make_notifier(s).push_text("hi")
    keys = {c[2]["X-Line-Retry-Key"] for c in s.calls}
    assert len(s.calls) == 3 and len(keys) == 1


def test_push_does_not_retry_4xx(monkeypatch):
    monkeypatch.setattr("kneo_alert.notifier.time.sleep", lambda _: None)
    s = FakeSession([FakeResp(401, "invalid token")])
    with pytest.raises(LineNotifyError, match="401"):
        make_notifier(s).push_text("hi")
    assert len(s.calls) == 1


def test_missing_credentials():
    with pytest.raises(LineNotifyError):
        LineMessagingNotifier(LineConfig(channel_access_token="", to=["U1"]))
    with pytest.raises(LineNotifyError):
        LineMessagingNotifier(LineConfig(channel_access_token="t", to=[]))


# ---------- Config ----------

def test_load_config_reads_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_TO", raising=False)
    (tmp_path / "c.yaml").write_text(
        "camera:\n  source: '1'\nroi:\n  polygon: [[0,0],[10,0],[10,10]]\n", encoding="utf-8")
    (tmp_path / ".env").write_text("LINE_CHANNEL_ACCESS_TOKEN=abc\nLINE_TO=U1, C2\n", encoding="utf-8")
    cfg = load_config(tmp_path / "c.yaml", tmp_path / ".env")
    assert cfg.camera.source == 1
    assert cfg.line.channel_access_token == "abc"
    assert cfg.line.to == ["U1", "C2"]


def test_load_config_rejects_unknown_key(tmp_path):
    (tmp_path / "c.yaml").write_text("trigger:\n  cooldown: 5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cooldown"):
        load_config(tmp_path / "c.yaml", tmp_path / "none.env")


def test_example_config_loads():
    from pathlib import Path
    cfg = load_config(Path(__file__).parent.parent / "config.example.yaml", "nonexistent.env")
    assert cfg.detector.type in ("hog", "yolo_onnx")


# ---------- Snapshot image (imgbb + LINE image message) ----------

class FakeJsonResp(FakeResp):
    def __init__(self, status, payload):
        super().__init__(status, str(payload))
        self._payload = payload

    def json(self):
        return self._payload


def test_push_with_image_sends_one_request_with_two_messages():
    s = FakeSession([FakeResp(200)])
    urls = [f"https://i.ibb.co/x/{i}.jpg" for i in range(6)]
    make_notifier(s).push("hi", image_urls=urls)
    assert len(s.calls) == 1  # one request -> counts as one message
    msgs = s.calls[0][1]["messages"]
    assert msgs[0] == {"type": "text", "text": "hi"}
    assert len(msgs) == 5  # text + at most 4 images
    assert msgs[1] == {"type": "image", "originalContentUrl": urls[0], "previewImageUrl": urls[0]}


def test_imgbb_upload_returns_https_url():
    from kneo_alert.notifier import ImgbbUploader

    class S:
        def post(self, url, params=None, data=None, timeout=None):
            self.url, self.params, self.data = url, params, data
            return FakeJsonResp(200, {"data": {"url": "https://i.ibb.co/abc/snap.jpg"}})
    sess = S()
    up = ImgbbUploader("KEY", expiration_sec=604800, session=sess)
    assert up.upload(b"\xff\xd8jpeg") == "https://i.ibb.co/abc/snap.jpg"
    assert sess.params == {"key": "KEY", "expiration": "604800"}
    assert "image" in sess.data


def test_async_notifier_falls_back_to_text_when_upload_fails():
    from kneo_alert.notifier import AsyncNotifier, LineNotifyError

    class BadUploader:
        def upload(self, jpeg):
            raise LineNotifyError("imgbb down")

    class Inner:
        def __init__(self):
            self.sent = []

        def push(self, text, image_urls=None):
            self.sent.append((text, image_urls))
    inner = Inner()
    n = AsyncNotifier(inner, BadUploader())
    n.send("alert", [b"jpeg1", b"jpeg2"])
    n.close()
    assert inner.sent == [("alert", [])]


# ---------- ROI editing ----------

def test_save_roi_polygon_keeps_comment(tmp_path):
    from kneo_alert.config import save_roi_polygon
    p = tmp_path / "c.yaml"
    p.write_text("roi:\n  polygon: []   # comment here\n  mode: foot\n", encoding="utf-8")
    save_roi_polygon(p, [[1, 2], [3.4, 4], [5, 6]])
    text = p.read_text(encoding="utf-8")
    assert "polygon: [[1, 2], [3, 4], [5, 6]]" in text and "# comment here" in text
    assert load_config(p, "none.env").roi.polygon == [[1, 2], [3, 4], [5, 6]]


def test_web_roi_api_roundtrip():
    import json
    import urllib.request
    from kneo_alert.stream import MjpegServer
    state = {"poly": [[0, 0], [10, 0], [10, 10]]}

    def set_roi(p):
        if len(p) < 3:
            raise ValueError("need 3 points")
        state["poly"] = p
    srv = MjpegServer(0, host="127.0.0.1", get_roi=lambda: state["poly"], set_roi=set_roi).start()
    base = f"http://127.0.0.1:{srv.port}"
    try:
        assert json.load(urllib.request.urlopen(base + "/api/roi"))["polygon"] == [[0, 0], [10, 0], [10, 10]]
        req = urllib.request.Request(base + "/api/roi", method="POST",
                                     data=json.dumps({"polygon": [[1, 1], [2, 1], [2, 2], [1, 2]]}).encode(),
                                     headers={"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req)) == {"ok": True}
        assert state["poly"] == [[1, 1], [2, 1], [2, 2], [1, 2]]
        bad = urllib.request.Request(base + "/api/roi", method="POST",
                                     data=b'{"polygon": [[1, 1]]}')
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
        assert b"<canvas" in urllib.request.urlopen(base + "/roi").read()
    finally:
        srv.stop()


def test_web_frame_endpoint_gets_raw_frame_from_main_loop():
    import threading
    import urllib.request
    from kneo_alert.stream import MjpegServer
    srv = MjpegServer(0, host="127.0.0.1").start()
    try:
        def feeder():  # stands in for the capture loop
            for _ in range(100):
                if srv.wants_raw:
                    srv.put_raw(np.zeros((48, 64, 3), np.uint8))
                    return
                threading.Event().wait(0.02)
        threading.Thread(target=feeder).start()
        data = urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/frame.jpg", timeout=5).read()
        assert data[:2] == b"\xff\xd8"
    finally:
        srv.stop()
