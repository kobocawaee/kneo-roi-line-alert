#!/usr/bin/env bash
# One-time setup on the board (Debian/Ubuntu-style Linux).
#
#   cd ~/kneo-roi-line-alert
#   bash deploy/setup_pi.sh                    # install venv + deps, run checks
#   bash deploy/setup_pi.sh --install-service  # also install & start the systemd service
set -euo pipefail

cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
INSTALL_SERVICE=0
[[ "${1:-}" == "--install-service" ]] && INSTALL_SERVICE=1

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!! %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mXX %s\033[0m\n' "$*"; exit 1; }

SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"
# pkg_install <debian-name> <arch-name>
pkg_install() {
  if command -v pacman >/dev/null; then $SUDO pacman -S --noconfirm --needed "$2"
  elif command -v apt-get >/dev/null; then $SUDO apt-get update && $SUDO apt-get install -y "$1"
  else die "Unknown package manager; install $1 manually"; fi
}

say "System"
uname -m
. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || true
command -v python3 >/dev/null || pkg_install python3 python
python3 --version

# ---------- venv + deps ----------
# Prefer the distro's OpenCV (matched to the system Python) when it's already installed;
# otherwise install an OpenCV wheel from pip.
say "Python virtual environment + dependencies"
python3 -m venv --help >/dev/null 2>&1 || pkg_install python3-venv python
PY="$APP_DIR/.venv/bin/python"
# Everything in requirements.txt except OpenCV (handled separately below).
REQ_NO_CV="$(mktemp)"; grep -viE '^\s*opencv' requirements.txt > "$REQ_NO_CV"
if python3 -c "import cv2" 2>/dev/null; then
  echo "Using system OpenCV $(python3 -c 'import cv2;print(cv2.__version__)')"
  [[ -x "$PY" ]] || python3 -m venv --system-site-packages .venv
  "$PY" -m pip install -q -r "$REQ_NO_CV"
else
  [[ -x "$PY" ]] || python3 -m venv .venv
  "$PY" -m pip install -q --upgrade pip
  "$PY" -m pip install -q -r "$REQ_NO_CV"
  # Headless boards don't need GUI OpenCV (and it pulls in libGL).
  if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then CV_PKG="opencv-python"; else CV_PKG="opencv-python-headless"; fi
  echo "Installing $CV_PKG from pip"
  if ! "$PY" -m pip install -q "$CV_PKG>=4.8,<5"; then
    warn "No pip wheel for $CV_PKG; installing the system OpenCV package instead"
    pkg_install python3-opencv python-opencv
    rm -rf .venv
    python3 -m venv --system-site-packages .venv
    "$PY" -m pip install -q -r "$REQ_NO_CV"
  fi
fi
"$PY" -c "import cv2, numpy, requests, yaml; print('OpenCV', cv2.__version__, '| numpy', numpy.__version__)"

# ---------- config ----------
say "Config files"
[[ -f config.yaml ]] || { cp config.example.yaml config.yaml; warn "Created config.yaml from example — edit it"; }
[[ -f .env ]] || { cp .env.example .env; warn "Created .env from example — fill in LINE token and LINE_TO"; }
grep -q '^LINE_CHANNEL_ACCESS_TOKEN=.\+' .env || warn "LINE_CHANNEL_ACCESS_TOKEN is empty in .env"
grep -q '^LINE_TO=.\+' .env || warn "LINE_TO is empty in .env"

# ---------- model ----------
say "Detector model"
DET_TYPE=$("$PY" -c "import yaml;print(yaml.safe_load(open('config.yaml'))['detector']['type'])")
MODEL=$("$PY" -c "import yaml;print(yaml.safe_load(open('config.yaml'))['detector']['model_path'])")
if [[ "$DET_TYPE" == "yolo_onnx" ]]; then
  [[ -f "$MODEL" ]] || die "$MODEL not found. Copy models/yolov8n.onnx from your PC."
  "$PY" - <<EOF
import time, numpy as np, sys
sys.path.insert(0, ".")
from kneo_alert.config import load_config
from kneo_alert.detector import create_detector
cfg = load_config("config.yaml", ".env")
d = create_detector(cfg.detector)
img = np.zeros((480, 640, 3), np.uint8)
d.detect(img); t = time.time()
for _ in range(3): d.detect(img)
print(f"YOLO OK, {(time.time()-t)/3*1000:.0f} ms per frame on this board")
EOF
else
  echo "detector.type=$DET_TYPE (no model file needed)"
fi

# ---------- camera ----------
say "Camera"
"$PY" - <<'EOF' || true
import sys, cv2, yaml
sys.path.insert(0, ".")
from kneo_alert.config import load_config
from kneo_alert.camera import open_camera
cfg = load_config("config.yaml", ".env").camera
cap = open_camera(cfg)
ok, frame = False, None
for _ in range(10):  # first frames can be empty while the sensor starts
    ok, frame = cap.read()
    if ok:
        break
cap.release()
if ok:
    cv2.imwrite("camera_test.jpg", frame)
    print(f"Camera source {cfg.source!r} OK, frame {frame.shape[1]}x{frame.shape[0]} -> camera_test.jpg")
else:
    print(f"!! Could not read from camera source {cfg.source!r}.")
EOF
[[ $EUID -eq 0 ]] || groups | grep -qw video || warn "User $(whoami) is not in the 'video' group: sudo usermod -aG video $(whoami) (then log out/in)"

# ---------- service ----------
if [[ $INSTALL_SERVICE == 1 ]]; then
  say "Installing systemd service"
  sed -e "s#^User=.*#User=$(whoami)#" \
      -e "s#^WorkingDirectory=.*#WorkingDirectory=$APP_DIR#" \
      -e "s#^ExecStart=.*#ExecStart=$PY -m kneo_alert#" \
      deploy/kneo-roi-alert.service | $SUDO tee /etc/systemd/system/kneo-roi-alert.service >/dev/null
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now kneo-roi-alert
  sleep 3
  $SUDO systemctl --no-pager status kneo-roi-alert | head -n 12
  echo "Logs: journalctl -u kneo-roi-alert -f"
fi

say "Done"
echo "Dry run:   $PY -m kneo_alert --dry-run"
echo "Real run:  $PY -m kneo_alert"
