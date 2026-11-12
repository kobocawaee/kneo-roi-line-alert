"""Tiny webhook receiver that prints the userId / groupId of incoming LINE events.

A group ID can only be obtained from a webhook event, so:
  1. Run:  python tools/line_webhook_id.py --port 8000
  2. Expose it over HTTPS, e.g.  ngrok http 8000   (or cloudflared tunnel --url http://localhost:8000)
  3. In LINE Developers console > Messaging API > Webhook URL, set  https://<tunnel-host>/callback
     and turn "Use webhook" on.
  4. Add the bot to the group (or send it a message 1:1). The IDs are printed here.
  5. Put the ID into LINE_TO in .env, then stop this script.

If LINE_CHANNEL_SECRET is set in .env, request signatures are verified.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kneo_alert.config import load_dotenv  # noqa: E402


def make_handler(secret: str):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if secret:
                sig = base64.b64encode(hmac.new(secret.encode(), body, hashlib.sha256).digest()).decode()
                if not hmac.compare_digest(sig, self.headers.get("X-Line-Signature", "")):
                    self.send_response(403)
                    self.end_headers()
                    print("! Rejected request with bad signature")
                    return
            self.send_response(200)
            self.end_headers()
            try:
                events = json.loads(body).get("events", [])
            except json.JSONDecodeError:
                return
            for ev in events:
                src = ev.get("source", {})
                kind = src.get("type")
                ident = src.get("groupId") or src.get("roomId") or src.get("userId")
                print(f"event={ev.get('type'):<10} source={kind:<6} id={ident}")
                if kind == "group":
                    print(f"  -> LINE_TO={src['groupId']}")

        def log_message(self, *_):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")
    secret = os.environ.get("LINE_CHANNEL_SECRET", "")
    if not secret:
        print("(LINE_CHANNEL_SECRET not set: signature check disabled)")
    print(f"Listening on http://0.0.0.0:{args.port}/callback  — Ctrl+C to stop")
    HTTPServer(("0.0.0.0", args.port), make_handler(secret)).serve_forever()


if __name__ == "__main__":
    main()
