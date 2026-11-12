"""Send one test message with the credentials in .env, to confirm LINE setup works.

Usage: python tools/send_test_message.py ["optional text"]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kneo_alert.config import LineConfig, load_dotenv  # noqa: E402
from kneo_alert.notifier import LineMessagingNotifier, LineNotifyError  # noqa: E402


def main():
    load_dotenv(ROOT / ".env")
    cfg = LineConfig(
        channel_access_token=os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", ""),
        to=[t.strip() for t in os.environ.get("LINE_TO", "").split(",") if t.strip()],
    )
    text = sys.argv[1] if len(sys.argv) > 1 else "✅ kneo-roi-line-alert 測試訊息"
    try:
        LineMessagingNotifier(cfg).push_text(text)
    except LineNotifyError as e:
        sys.exit(f"Failed: {e}")
    print(f"Sent to {len(cfg.to)} recipient(s)")


if __name__ == "__main__":
    main()
