"""Main capture -> detect -> ROI -> trigger -> LINE loop."""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from .camera import open_camera as _open_camera
from .config import AppConfig, load_config, save_roi_polygon, validate_polygon
from .detector import create_detector
from .notifier import AsyncNotifier, DryRunNotifier, ImgbbUploader, LineMessagingNotifier
from .roi import Roi
from .trigger import EntryTrigger

log = logging.getLogger("kneo_alert")

# LINE allows 5 message objects per push: 1 text + up to 4 images (still counts as 1 message).
MAX_IMAGES = 4


def open_camera(cfg: AppConfig):
    return _open_camera(cfg.camera)


def format_message(template: str, count: int) -> str:
    return template.format(count=count, time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def annotate(frame, roi: Roi, dets, inside) -> None:
    roi.draw(frame)
    inside_ids = {id(d) for d in inside}
    for d in dets:
        color = (0, 0, 255) if id(d) in inside_ids else (0, 255, 0)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.circle(frame, d.foot_point, 4, color, -1)
        cv2.putText(frame, f"{d.score:.2f}", (d.x1, max(12, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


class DetectionWorker:
    """Runs detect -> ROI -> trigger -> alert.

    async mode (live cameras): a background thread always works on the newest submitted
    frame and drops older ones, so the capture/preview loop never waits for the detector.
    sync mode (video files): every submitted frame is processed inline, deterministically.
    """

    def __init__(self, cfg: AppConfig, detector, roi: Roi, trigger: EntryTrigger, notifier,
                 snapshot_dir: Path | None, sync: bool, frame_size: tuple[int, int]):
        self.cfg, self.detector, self.roi, self.trigger = cfg, detector, roi, trigger
        self.notifier, self.snapshot_dir, self.sync = notifier, snapshot_dir, sync
        self.frame_size = frame_size
        self._events: list = []  # one snapshot per arrival event in the current visit
        self.alerts = 0
        self.processed = 0
        self.dets: list = []
        self.inside: list = []
        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._stop = False
        self._error: BaseException | None = None
        self._thread = None
        if not sync:
            self._thread = threading.Thread(target=self._loop, name="detector", daemon=True)
            self._thread.start()

    def submit(self, frame) -> None:
        if self._error:
            raise RuntimeError("detector thread failed") from self._error
        if self.sync:
            self._process(frame)
            return
        with self._lock:
            self._pending = frame  # replaces any frame the detector hasn't picked up yet
        self._wake.set()

    def latest(self):
        with self._lock:
            return self.dets, self.inside

    def close(self) -> None:
        self._stop = True
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self) -> None:
        try:
            while not self._stop:
                self._wake.wait()
                self._wake.clear()
                with self._lock:
                    frame, self._pending = self._pending, None
                if frame is not None and not self._stop:
                    self._process(frame)
        except BaseException as e:  # surfaced to the main loop on the next submit()
            log.exception("Detector thread crashed")
            self._error = e

    def set_roi(self, polygon) -> None:
        """Swap in a new ROI polygon (called from the web UI thread)."""
        self.cfg.roi.polygon = [list(p) for p in polygon]
        self.roi = Roi(self.cfg.roi, self.frame_size)
        log.info("ROI updated: %s", self.cfg.roi.polygon)

    def _process(self, frame) -> None:
        roi = self.roi  # may be swapped by set_roi() from another thread
        dets = self.detector.detect(frame)
        inside = roi.filter(dets)
        with self._lock:
            self.dets, self.inside = dets, inside
        self.processed += 1

        res = self.trigger.update(len(inside))
        if res.arrived:
            snap = (frame.copy() if self.sync else frame, roi, dets, inside)
            if res.same_group and self._events:
                # More people joined those still in the ROI: one picture of the whole group.
                self._events[-1] = snap
            elif len(self._events) < MAX_IMAGES:
                self._events.append(snap)
            else:
                log.info("More than %d arrival events; extra snapshots not attached", MAX_IMAGES)
        if res:
            self._alert(res.count)

    def _alert(self, count: int) -> None:
        self.alerts += 1
        msg = format_message(self.cfg.alert.message, count)
        log.info("ALERT: %d person(s) entered ROI (%d snapshot(s))", count, len(self._events))
        jpegs: list[bytes] = []
        if self.snapshot_dir is not None or self.cfg.alert.attach_image:
            stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
            for i, (frame, roi, dets, inside) in enumerate(self._events, 1):
                snap = frame.copy()
                annotate(snap, roi, dets, inside)
                ok, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                jpegs.append(buf.tobytes())
                if self.snapshot_dir is not None:
                    path = self.snapshot_dir / f"alert_{stamp}_{i}.jpg"
                    path.write_bytes(jpegs[-1])
                    log.info("Snapshot saved: %s", path)
        self._events = []
        self.notifier.send(msg, jpegs if self.cfg.alert.attach_image else [])


def run(cfg: AppConfig, dry_run: bool = False, show: bool = False, max_frames: int | None = None,
        stream_port: int | None = None) -> int:
    detector = create_detector(cfg.detector)
    trigger = EntryTrigger(cfg.trigger)
    inner = DryRunNotifier() if dry_run else LineMessagingNotifier(cfg.line)
    uploader = None
    if cfg.alert.attach_image and not dry_run:
        if cfg.imgbb_api_key:
            uploader = ImgbbUploader(cfg.imgbb_api_key, cfg.alert.image_expiration_sec)
        else:
            log.warning("alert.attach_image is on but IMGBB_API_KEY is not set in .env — "
                        "sending text only")
    notifier = AsyncNotifier(inner, uploader)

    snapshot_dir = None
    if cfg.alert.save_snapshot:
        snapshot_dir = Path(cfg.alert.snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    stop = False

    def _sigterm(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _sigterm)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _sigterm)

    worker: DetectionWorker | None = None

    def web_get_roi():
        return cfg.roi.polygon

    def web_set_roi(points):
        if worker is None:
            raise ValueError("camera not ready yet")
        w, h = worker.frame_size
        pts = [[min(max(x, 0), w - 1), min(max(y, 0), h - 1)] for x, y in validate_polygon(points)]
        worker.set_roi(pts)
        if cfg.path is not None:
            save_roi_polygon(cfg.path, pts)
            log.info("ROI saved to %s", cfg.path)

    streamer = None
    if stream_port:
        from .stream import MjpegServer
        streamer = MjpegServer(stream_port, get_roi=web_get_roi, set_roi=web_set_roi).start()

    cap = open_camera(cfg)
    frame_idx = 0
    is_file = isinstance(cfg.camera.source, str) and Path(cfg.camera.source).is_file()
    fps_t0, fps_n, fps = time.monotonic(), 0, 0.0
    if is_file:
        # Video files are processed as fast as possible, so time the gather window/cooldown
        # by the video's own timeline instead of the wall clock.
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        trigger.clock = lambda: frame_idx / video_fps

    log.info("Started (detector=%s, dry_run=%s)", cfg.detector.type, dry_run)
    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                if is_file:
                    log.info("End of video file")
                    break
                log.warning("Camera read failed; reconnecting in %.1fs", cfg.camera.reconnect_delay_sec)
                cap.release()
                time.sleep(cfg.camera.reconnect_delay_sec)
                cap = open_camera(cfg)
                continue

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            if worker is None:
                h, w = frame.shape[:2]
                log.info("Frame size %dx%d", w, h)
                worker = DetectionWorker(cfg, detector, Roi(cfg.roi, (w, h)), trigger, notifier,
                                         snapshot_dir, sync=is_file, frame_size=(w, h))

            if frame_idx % max(1, cfg.camera.process_every_n_frames) == 0:
                worker.submit(frame if worker.sync else frame.copy())

            if streamer and streamer.wants_raw:
                streamer.put_raw(frame)  # un-annotated frame for the ROI editor page

            if show or (streamer and streamer.has_clients):
                fps_n += 1
                now = time.monotonic()
                if now - fps_t0 >= 1.0:
                    fps, fps_t0, fps_n = fps_n / (now - fps_t0), now, 0
                dets, inside = worker.latest()
                annotate(frame, worker.roi, dets, inside)
                status = (f"COUNTING {trigger.total}" if trigger.gathering
                          else "OCCUPIED" if trigger.occupied else "empty")
                cv2.putText(frame, f"ROI: {len(inside)} ({status})  {fps:.0f} fps", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if streamer:
                    streamer.update(frame)
                if show:
                    cv2.imshow("kneo-roi-line-alert", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
    finally:
        if worker:
            worker.close()
        cap.release()
        if show:
            cv2.destroyAllWindows()
        if streamer:
            streamer.stop()
        notifier.close()
        alerts = worker.alerts if worker else 0
        log.info("Stopped after %d frames (%d detected), %d alert(s)",
                 frame_idx, worker.processed if worker else 0, alerts)
    return alerts


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Kneo Pi ROI person-entry LINE alert")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--env", default=".env")
    ap.add_argument("--dry-run", action="store_true", help="log the message instead of sending to LINE")
    ap.add_argument("--show", action="store_true", help="show a preview window (needs a display)")
    ap.add_argument("--stream", type=int, metavar="PORT", nargs="?", const=8080,
                    help="serve a live view at http://<ip>:PORT/ (default 8080); LAN only")
    ap.add_argument("--source", help="override camera.source (index, file, or URL)")
    ap.add_argument("--max-frames", type=int, help="stop after N frames (for testing)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config, args.env)
    if args.source is not None:
        cfg.camera.source = int(args.source) if args.source.isdigit() else args.source
    run(cfg, dry_run=args.dry_run, show=args.show, max_frames=args.max_frames, stream_port=args.stream)
