#!/usr/bin/env bash
# Build OpenNI2 + PS1080 driver (Xtion / PrimeSense) on aarch64 Linux and install to /opt/OpenNI2.
#
#   bash deploy/build_openni2.sh
#
# Upstream OpenNI2 (structureio/OpenNI2, last commit 2019) only knows x86/x64/32-bit ARM.
# This script patches it for aarch64:
#   1. CommonDefs.mak : map `uname -m` = aarch64 to a new platform "Arm64"
#   2. Platform.Arm64 : plain -O2 flags (Platform.Arm hardcodes armv7/NEON flags that fail on aarch64)
#   3. OniPlatform.h / XnPlatform.h : accept __aarch64__ as Linux-Arm
#   4. ALLOW_WARNINGS=1 : drop -Werror (modern GCC warns a lot on this old code)
# Only the pieces we need are built: XnLib, OpenNI core, DepthUtils, PS1080 driver.
set -euo pipefail

SRC_DIR="${OPENNI2_SRC:-/opt/OpenNI2-src}"
PREFIX="${OPENNI2_PREFIX:-/opt/OpenNI2}"
SUDO=""; [[ $EUID -ne 0 ]] && SUDO="sudo"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }

say "Build dependencies"
if command -v pacman >/dev/null; then
  # Only install what's missing. Never upgrade already-installed packages: on a vendor
  # image with a stale package DB that hits 404s, and partial upgrades can break Arch.
  missing=()
  command -v g++  >/dev/null || missing+=(gcc)
  command -v make >/dev/null || missing+=(make)
  command -v git  >/dev/null || command -v curl >/dev/null || missing+=(git)
  [[ -f /usr/include/libusb-1.0/libusb.h ]] || missing+=(libusb)
  if (( ${#missing[@]} )); then
    echo "Installing: ${missing[*]}"
    $SUDO pacman -S --noconfirm "${missing[@]}" || {
      echo "pacman failed. If you see 404 errors, the package DB is stale: run 'pacman -Sy' and retry."
      exit 1; }
  else
    echo "g++, make, libusb headers and git/curl already present"
  fi
else
  $SUDO apt-get update && $SUDO apt-get install -y build-essential git libusb-1.0-0-dev
fi

say "Source ($SRC_DIR)"
if [[ ! -f "$SRC_DIR/Makefile" ]]; then
  $SUDO mkdir -p "$SRC_DIR"
  if command -v git >/dev/null; then
    $SUDO rm -rf "$SRC_DIR"
    $SUDO git clone --depth 1 https://github.com/structureio/OpenNI2.git "$SRC_DIR"
  else
    curl -fL https://github.com/structureio/OpenNI2/archive/refs/heads/master.tar.gz \
      | $SUDO tar -xz --strip-components=1 -C "$SRC_DIR"
  fi
  [[ $EUID -ne 0 ]] && $SUDO chown -R "$(id -u):$(id -g)" "$SRC_DIR"
fi
cd "$SRC_DIR"

say "Patching for aarch64"
B=ThirdParty/PSCommon/BuildSystem
if ! grep -q aarch64 $B/CommonDefs.mak; then
  sed -i 's/^else ifneq (,$(findstring arm,$(MACHINE)))$/else ifneq (,$(findstring aarch64,$(MACHINE)))\n\tHOST_PLATFORM = Arm64\n&/' $B/CommonDefs.mak
fi
grep -q "HOST_PLATFORM = Arm64" $B/CommonDefs.mak || { echo "CommonDefs.mak patch failed"; exit 1; }

cat > $B/Platform.Arm64 <<'EOF'
# aarch64: no armv7/NEON-specific flags; let the compiler pick generic armv8 code.
CFLAGS += -O2 -fno-strict-aliasing -std=gnu++14
EOF

for h in Include/OniPlatform.h ThirdParty/PSCommon/XnLib/Include/XnPlatform.h; do
  sed -i 's/#elif (__linux__ \&\& __arm__)$/#elif (__linux__ \&\& (__arm__ || __aarch64__))/' "$h"
  grep -q "__aarch64__" "$h" || { echo "Header patch failed: $h"; exit 1; }
done

say "Compiling (takes a few minutes)"
make -j"$(nproc)" ALLOW_WARNINGS=1 \
  ThirdParty/PSCommon/XnLib/Source \
  Source/Core \
  Source/DepthUtils \
  Source/Drivers/PS1080

OUT="Bin/Arm64-Release"
ls -la "$OUT/libOpenNI2.so" "$OUT/OpenNI2/Drivers/libPS1080.so"

say "Installing to $PREFIX"
$SUDO mkdir -p "$PREFIX/OpenNI2/Drivers"
$SUDO cp "$OUT/libOpenNI2.so" "$PREFIX/"
$SUDO cp "$OUT/OpenNI2/Drivers/libPS1080.so" "$PREFIX/OpenNI2/Drivers/"
$SUDO cp Config/OpenNI.ini "$PREFIX/"
$SUDO cp Config/OpenNI2/Drivers/PS1080.ini "$PREFIX/OpenNI2/Drivers/"

# Let non-root users open the device (root works regardless).
if [[ -d /etc/udev/rules.d ]]; then
  echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1d27", MODE="0666"' | $SUDO tee /etc/udev/rules.d/55-primesense.rules >/dev/null
  $SUDO udevadm control --reload-rules 2>/dev/null || true
fi

say "Done"
echo "OpenNI2 installed in $PREFIX"
echo "Next: test with  python deploy/check_openni2.py"
