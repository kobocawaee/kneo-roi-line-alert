"""Load config.yaml and secrets from environment / .env."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .camera import CameraConfig  # camera settings live with the camera code (differs per version)


@dataclass
class DetectorConfig:
    # "kneron_nef" (KL730 NPU), "cut_onnx" (same models on CPU, for PC tests), "yolo_onnx" or "hog"
    type: str = "hog"
    model_path: str = "models/yolov8n.onnx"
    input_size: int = 640
    conf_threshold: float = 0.45
    nms_threshold: float = 0.45
    # --- kneron_nef / cut_onnx ---
    nef_path: str = "models/models_730.nef"
    firmware_path: str = "models/kp_firmware.tar"
    person_model_id: int = 32769
    face_model_id: int = 32770
    person_onnx: str = "npu/yolov8n_320_cut.onnx"   # cut_onnx only
    face_onnx: str = "npu/ultraface_320_cut.onnx"   # cut_onnx only
    require_face: bool = False  # count a person only if a face is seen in the upper half of the box
    face_conf: float = 0.7


@dataclass
class RoiConfig:
    # Polygon vertices in pixel coordinates of the processed frame.
    polygon: list[list[int]] = field(default_factory=list)
    # "foot": bottom-center of bbox must be inside; "overlap": bbox overlap ratio.
    mode: str = "foot"
    min_overlap: float = 0.3


@dataclass
class TriggerConfig:
    enter_frames: int = 3  # consecutive detections with someone in ROI before it counts as entered
    exit_sec: float = 4.0   # nobody in the ROI for this long -> it counts as empty again
    leave_sec: float = 1.0  # fewer people for this long -> someone really left (not a flicker)
    exit_frames: int | None = None   # deprecated (frame-based, depended on detector speed); ignored
    leave_frames: int | None = None  # deprecated; ignored
    cooldown_sec: float = 60.0
    # After the first person enters, keep counting arrivals (together or one after another)
    # for this long, then send ONE message with the total. 0 = alert immediately.
    gather_sec: float = 5.0


@dataclass
class AlertConfig:
    message: str = "⚠️ 偵測到有人進入監控區域（{count} 人）\n時間：{time}"
    snapshot_dir: str = "snapshots"
    save_snapshot: bool = True
    # Send the annotated snapshot with the LINE message (needs IMGBB_API_KEY in .env).
    attach_image: bool = False
    image_expiration_sec: int = 604800  # imgbb auto-delete; 0 = keep forever


@dataclass
class LineConfig:
    channel_access_token: str = ""
    to: list[str] = field(default_factory=list)  # user / group / room IDs
    timeout_sec: float = 10.0


@dataclass
class AppConfig:
    camera: CameraConfig
    detector: DetectorConfig
    roi: RoiConfig
    trigger: TriggerConfig
    alert: AlertConfig
    line: LineConfig
    imgbb_api_key: str = ""
    path: Path | None = None  # where config.yaml was loaded from (for saving ROI edits)


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Does not override existing env vars."""
    p = Path(path)
    if not p.is_file():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _section(cls, data: dict | None):
    data = data or {}
    known = {f for f in cls.__dataclass_fields__}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys in [{cls.__name__}]: {sorted(unknown)}")
    return cls(**data)


def validate_polygon(points) -> list[list[int]]:
    pts = [[int(round(float(x))), int(round(float(y)))] for x, y in points]
    if len(pts) < 3:
        raise ValueError("roi.polygon needs at least 3 points")
    return pts


def load_config(path: str | Path = "config.yaml", env_file: str | Path = ".env") -> AppConfig:
    load_dotenv(env_file)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = AppConfig(
        camera=_section(CameraConfig, raw.get("camera")),
        detector=_section(DetectorConfig, raw.get("detector")),
        roi=_section(RoiConfig, raw.get("roi")),
        trigger=_section(TriggerConfig, raw.get("trigger")),
        alert=_section(AlertConfig, raw.get("alert")),
        line=LineConfig(),
        path=Path(path),
    )

    cfg.line.channel_access_token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    to = os.environ.get("LINE_TO", "")
    cfg.line.to = [t.strip() for t in to.split(",") if t.strip()]
    cfg.imgbb_api_key = os.environ.get("IMGBB_API_KEY", "")

    # A numeric string like "0" means a local camera index.
    if isinstance(cfg.camera.source, str) and cfg.camera.source.isdigit():
        cfg.camera.source = int(cfg.camera.source)

    for old, new in (("exit_frames", "exit_sec"), ("leave_frames", "leave_sec")):
        if getattr(cfg.trigger, old) is not None:
            logging.getLogger(__name__).warning(
                "trigger.%s is no longer used (it depended on detector speed); using trigger.%s = %s s",
                old, new, getattr(cfg.trigger, new))

    if cfg.roi.polygon:
        cfg.roi.polygon = validate_polygon(cfg.roi.polygon)
    if cfg.roi.mode not in ("foot", "overlap"):
        raise ValueError("roi.mode must be 'foot' or 'overlap'")
    return cfg


def save_roi_polygon(config_path: str | Path, points) -> None:
    """Replace the `  polygon:` line under roi in config.yaml, keeping comments intact."""
    p = Path(config_path)
    pts = validate_polygon(points)
    text = p.read_text(encoding="utf-8")
    text, n = re.subn(r"(?m)^(  polygon:)[^#\n]*(#.*)?$",
                      lambda m: f"{m.group(1)} {pts}" + (f"  {m.group(2)}" if m.group(2) else ""),
                      text, count=1)
    if n == 0:
        raise ValueError(f"Could not find '  polygon:' under roi in {p}")
    p.write_text(text, encoding="utf-8")
