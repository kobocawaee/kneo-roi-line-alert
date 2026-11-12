"""Tiny web UI on the board, viewable in any browser:

  http://<board-ip>:<port>/      live MJPEG view (ROI + detections drawn on it)
  http://<board-ip>:<port>/roi   click on a still frame to redraw the ROI; saving applies it
                                 immediately and writes it to config.yaml

No auth and no TLS: only use it on a trusted LAN, never expose the port to the internet
(anyone who can reach it can watch the camera and change the ROI).
JPEG encoding for the live view only happens while at least one browser is watching.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kneo-roi-line-alert</title>
<style>body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center}
a{color:#7cf}img{max-width:100%;height:auto;margin-top:8px}</style></head>
<body><div>即時畫面 · <a href="/roi">修改偵測範圍（ROI）</a></div><img src="/stream"></body></html>
""".encode("utf-8")

ROI_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>修改 ROI</title>
<style>
body{margin:0;background:#111;color:#ddd;font-family:sans-serif;text-align:center;padding:8px}
#wrap{position:relative;display:inline-block;max-width:100%}
#img{display:block;max-width:100%;height:auto}
#cv{position:absolute;left:0;top:0;width:100%;height:100%;cursor:crosshair;touch-action:none}
button{font-size:16px;margin:6px 4px;padding:8px 14px;border-radius:6px;border:0;cursor:pointer}
#save{background:#2a7;color:#fff}#msg{min-height:1.4em;margin:6px}a{color:#7cf}
</style></head><body>
<div>點畫面加點（至少 3 點）圍出偵測範圍，按「儲存」立即生效 · <a href="/">回即時畫面</a></div>
<div id="wrap"><img id="img"><canvas id="cv"></canvas></div>
<div>
<button id="undo">復原</button><button id="clear">清除</button>
<button id="reload">重新擷取畫面</button><button id="save">儲存</button>
</div><div id="msg"></div>
<script>
const img=document.getElementById('img'),cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const msg=document.getElementById('msg');let pts=[],W=0,H=0;
function draw(){cv.width=img.naturalWidth;cv.height=img.naturalHeight;ctx.clearRect(0,0,cv.width,cv.height);
 if(!pts.length)return;ctx.lineWidth=Math.max(2,cv.width/320);ctx.strokeStyle='#ffc800';ctx.fillStyle='rgba(255,200,0,.25)';
 ctx.beginPath();pts.forEach((p,i)=>i?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));
 if(pts.length>2){ctx.closePath();ctx.fill();}ctx.stroke();
 ctx.fillStyle='#f33';pts.forEach(p=>{ctx.beginPath();ctx.arc(p[0],p[1],ctx.lineWidth*2,0,7);ctx.fill();});}
function grab(){msg.textContent='擷取畫面中…';img.src='/frame.jpg?t='+Date.now();}
img.onload=()=>{msg.textContent='';draw();};img.onerror=()=>{msg.textContent='擷取畫面失敗，稍後再試';};
cv.addEventListener('pointerdown',e=>{const r=cv.getBoundingClientRect();
 pts.push([Math.round((e.clientX-r.left)*img.naturalWidth/r.width),Math.round((e.clientY-r.top)*img.naturalHeight/r.height)]);draw();});
document.getElementById('undo').onclick=()=>{pts.pop();draw();};
document.getElementById('clear').onclick=()=>{pts=[];draw();};
document.getElementById('reload').onclick=grab;
document.getElementById('save').onclick=async()=>{
 if(pts.length<3){msg.textContent='至少要 3 個點';return;}
 const r=await fetch('/api/roi',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygon:pts})});
 const j=await r.json().catch(()=>({}));msg.textContent=r.ok?'已儲存並生效 ✔':'儲存失敗：'+(j.error||r.status);};
fetch('/api/roi').then(r=>r.json()).then(j=>{pts=j.polygon||[];grab();}).catch(grab);
</script></body></html>
""".encode("utf-8")


class MjpegServer:
    def __init__(self, port: int, host: str = "0.0.0.0", quality: int = 70,
                 get_roi: Callable[[], list] | None = None,
                 set_roi: Callable[[list], None] | None = None):
        self.quality = quality
        self.get_roi = get_roi
        self.set_roi = set_roi
        self._jpeg: bytes | None = None
        self._seq = 0
        self._cond = threading.Condition()
        self._clients = 0
        self._stopped = False
        self._want_raw = threading.Event()
        self._raw_ready = threading.Event()
        self._raw_jpeg: bytes | None = None
        self.httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, name="mjpeg", daemon=True)

    def start(self) -> "MjpegServer":
        self.thread.start()
        log.info("Live view: http://<this-device-ip>:%d/   ROI editor: http://<this-device-ip>:%d/roi",
                 self.port, self.port)
        return self

    @property
    def has_clients(self) -> bool:
        return self._clients > 0

    @property
    def wants_raw(self) -> bool:
        return self._want_raw.is_set()

    def put_raw(self, frame: np.ndarray) -> None:
        """Main loop hands over an un-annotated frame when the ROI page asked for one."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            self._raw_jpeg = buf.tobytes()
            self._want_raw.clear()
            self._raw_ready.set()

    def update(self, frame: np.ndarray) -> None:
        if not self.has_clients:
            return
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._cond.notify_all()

    def stop(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _json(self, code: int, obj) -> None:
                self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    self._send(200, PAGE, "text/html; charset=utf-8")
                elif path == "/stream":
                    self._stream()
                elif path == "/roi":
                    self._send(200, ROI_PAGE, "text/html; charset=utf-8")
                elif path == "/frame.jpg":
                    server._raw_ready.clear()
                    server._want_raw.set()
                    if server._raw_ready.wait(timeout=5) and server._raw_jpeg:
                        self._send(200, server._raw_jpeg, "image/jpeg")
                    else:
                        self.send_error(503, "No camera frame available")
                elif path == "/api/roi":
                    self._json(200, {"polygon": server.get_roi() if server.get_roi else []})
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path.split("?", 1)[0] != "/api/roi" or server.set_roi is None:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    if length > 100_000:
                        raise ValueError("request too large")
                    data = json.loads(self.rfile.read(length) or b"{}")
                    server.set_roi(data["polygon"])
                except Exception as e:
                    self._json(400, {"error": str(e)})
                    return
                self._json(200, {"ok": True})

            def _stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with server._cond:
                    server._clients += 1
                last = -1
                try:
                    while True:
                        with server._cond:
                            server._cond.wait_for(
                                lambda: server._stopped or server._seq != last, timeout=5)
                            if server._stopped:
                                return
                            if server._seq == last or server._jpeg is None:
                                continue
                            last, jpeg = server._seq, server._jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass
                finally:
                    with server._cond:
                        server._clients -= 1

        return Handler
