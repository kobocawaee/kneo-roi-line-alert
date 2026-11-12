"""LINE Messaging API push notifier, with optional snapshot image via imgbb.

LINE Notify was discontinued on 2025-03-31; this uses the Messaging API
push endpoint instead: https://developers.line.biz/en/reference/messaging-api/#send-push-message

LINE image messages must point at a public HTTPS URL (the API can't take an upload),
so snapshots are first uploaded to imgbb (https://api.imgbb.com/). The text and up to 4
images go out in a single push request (5 message objects max), which counts as one message.
"""

from __future__ import annotations

import base64
import logging
import queue
import threading
import time
import uuid

import requests

from .config import LineConfig

log = logging.getLogger(__name__)

PUSH_URL = "https://api.line.me/v2/bot/message/push"
IMGBB_URL = "https://api.imgbb.com/1/upload"
MAX_TEXT_LEN = 5000
MAX_IMAGES = 4  # 5 message objects per push, one is the text


class LineNotifyError(RuntimeError):
    pass


class LineMessagingNotifier:
    def __init__(self, cfg: LineConfig, session: requests.Session | None = None, max_retries: int = 3):
        if not cfg.channel_access_token:
            raise LineNotifyError("LINE_CHANNEL_ACCESS_TOKEN is not set (put it in .env)")
        if not cfg.to:
            raise LineNotifyError("LINE_TO is not set (user/group ID, comma-separated for several)")
        self.cfg = cfg
        self.session = session or requests.Session()
        self.max_retries = max_retries

    def push_text(self, text: str) -> None:
        self.push(text)

    def push(self, text: str, image_urls: list[str] | None = None) -> None:
        """Send text (+ up to 4 images) to every recipient in cfg.to. Raises LineNotifyError on failure."""
        messages = [{"type": "text", "text": text[:MAX_TEXT_LEN]}]
        for url in (image_urls or [])[:MAX_IMAGES]:
            messages.append({"type": "image", "originalContentUrl": url, "previewImageUrl": url})
        errors = []
        for to in self.cfg.to:
            try:
                self._push(to, messages)
            except LineNotifyError as e:
                errors.append(f"{to[:6]}…: {e}")
        if errors:
            raise LineNotifyError("; ".join(errors))

    def _push(self, to: str, messages: list[dict]) -> None:
        # Same retry key across retries => LINE won't deliver duplicates.
        retry_key = str(uuid.uuid4())
        headers = {
            "Authorization": f"Bearer {self.cfg.channel_access_token}",
            "Content-Type": "application/json",
            "X-Line-Retry-Key": retry_key,
        }
        body = {"to": to, "messages": messages}
        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(PUSH_URL, json=body, headers=headers, timeout=self.cfg.timeout_sec)
            except requests.RequestException as e:
                last_err = f"network error: {e}"
            else:
                if resp.status_code == 200:
                    return
                if resp.status_code == 409:  # already accepted with this retry key
                    return
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                # 4xx other than 429 won't succeed on retry (bad token, bad ID, quota used up...)
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    break
            if attempt < self.max_retries:
                time.sleep(2 ** (attempt - 1))
        raise LineNotifyError(last_err)


class ImgbbUploader:
    """Uploads a JPEG to imgbb and returns its public HTTPS URL."""

    def __init__(self, api_key: str, expiration_sec: int = 0, timeout_sec: float = 20.0,
                 session: requests.Session | None = None):
        if not api_key:
            raise LineNotifyError("IMGBB_API_KEY is not set (put it in .env)")
        self.api_key = api_key
        self.expiration_sec = expiration_sec
        self.timeout_sec = timeout_sec
        self.session = session or requests.Session()

    def upload(self, jpeg: bytes) -> str:
        params = {"key": self.api_key}
        if self.expiration_sec:
            # imgbb accepts 60 s .. 180 days
            params["expiration"] = str(min(max(int(self.expiration_sec), 60), 15552000))
        try:
            resp = self.session.post(IMGBB_URL, params=params,
                                     data={"image": base64.b64encode(jpeg).decode("ascii")},
                                     timeout=self.timeout_sec)
        except requests.RequestException as e:
            raise LineNotifyError(f"imgbb network error: {e}") from e
        if resp.status_code != 200:
            raise LineNotifyError(f"imgbb HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            url = resp.json()["data"]["url"]
        except (ValueError, KeyError, TypeError) as e:
            raise LineNotifyError(f"imgbb unexpected response: {resp.text[:300]}") from e
        if not str(url).startswith("https://"):
            raise LineNotifyError(f"imgbb returned a non-HTTPS URL: {url}")
        return url


class DryRunNotifier:
    """Prints instead of sending. Used with --dry-run."""

    def push_text(self, text: str) -> None:
        self.push(text)

    def push(self, text: str, image_urls: list[str] | None = None) -> None:
        log.info("[DRY RUN] LINE message:\n%s", text)


class AsyncNotifier:
    """Sends in a background thread so network latency never stalls the camera loop."""

    def __init__(self, inner, uploader: ImgbbUploader | None = None, maxsize: int = 20):
        self.inner = inner
        self.uploader = uploader
        self.q: queue.Queue[tuple[str, list[bytes]] | None] = queue.Queue(maxsize=maxsize)
        self.thread = threading.Thread(target=self._run, name="line-notifier", daemon=True)
        self.thread.start()

    def push_text(self, text: str) -> None:
        self.send(text)

    def send(self, text: str, images: list[bytes] | None = None) -> None:
        """Queue a message; JPEG images (if an uploader is set) are attached as photos."""
        try:
            self.q.put_nowait((text, list(images or [])[:MAX_IMAGES]))
        except queue.Full:
            log.warning("Notification queue full; dropping message")

    def _run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            text, images = item
            urls: list[str] = []
            if images:
                if self.uploader is not None:
                    for jpeg in images:
                        try:
                            urls.append(self.uploader.upload(jpeg))
                            log.info("Snapshot uploaded: %s", urls[-1])
                        except Exception as e:  # still send the text (and other images)
                            log.error("Snapshot upload failed: %s", e)
                elif isinstance(self.inner, DryRunNotifier):
                    log.info("[DRY RUN] %d snapshot(s) would be attached (%s KB)", len(images),
                             ", ".join(str(len(j) // 1024) for j in images))
            try:
                self.inner.push(text, urls)
                if not isinstance(self.inner, DryRunNotifier):
                    log.info("LINE message sent")
            except Exception as e:  # keep the worker alive
                log.error("LINE push failed: %s", e)

    def close(self, timeout: float = 15.0) -> None:
        self.q.put(None)
        self.thread.join(timeout)
