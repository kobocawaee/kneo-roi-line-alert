"""camera.source: auto / /dev paths (webcam version only)."""

import pytest

from kneo_alert import camera


def make_sysfs(tmp_path, devices):
    for num, (name, index) in devices.items():
        d = tmp_path / f"video{num}"
        d.mkdir()
        (d / "name").write_text(name + "\n")
        (d / "index").write_text(f"{index}\n")
    return tmp_path


VPL = {n: (f"vpl_voc2-{n // 4}-{n % 4}", n % 4) for n in range(8)}


def test_auto_finds_webcam_after_display_outputs(tmp_path):
    """Boot 1 on the KNEO Pi: vpl_voc outputs video0-7, webcam video8 (+ metadata video9)."""
    devs = dict(VPL)
    devs[8] = ("GENERAL WEBCAM: GENERAL WEBCAM", 0)
    devs[9] = ("GENERAL WEBCAM: GENERAL WEBCAM", 1)
    assert camera.find_webcam_index(make_sysfs(tmp_path, devs)) == 8


def test_auto_finds_webcam_when_it_enumerates_first(tmp_path):
    """Boot 2: webcam became video0/1, vpl_voc moved to video2-9."""
    devs = {0: ("GENERAL WEBCAM: GENERAL WEBCAM", 0), 1: ("GENERAL WEBCAM: GENERAL WEBCAM", 1)}
    devs.update({n + 2: v for n, v in VPL.items()})
    assert camera.find_webcam_index(make_sysfs(tmp_path, devs)) == 0


def test_auto_without_camera(tmp_path):
    assert camera.find_webcam_index(make_sysfs(tmp_path, dict(VPL))) is None


def test_dev_path_resolves_to_index(monkeypatch):
    monkeypatch.setattr(camera.os.path, "realpath", lambda p: "/dev/video8")
    assert camera.resolve_source("/dev/v4l/by-id/usb-GENERAL_WEBCAM-video-index0") == 8
    monkeypatch.setattr(camera.os.path, "realpath", lambda p: "/dev/media0")
    with pytest.raises(RuntimeError):
        camera.resolve_source("/dev/v4l/by-id/whatever")


def test_other_sources_pass_through():
    assert camera.resolve_source(2) == 2
    assert camera.resolve_source("rtsp://cam/stream") == "rtsp://cam/stream"
