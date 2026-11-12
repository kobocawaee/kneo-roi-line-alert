"""Copy shared code from the webcam version (the one you hand in) to the Xtion version.

    python sync_versions.py            # show what would change
    python sync_versions.py --apply    # copy it

Workflow: add features in kneo-roi-line-alert/ (webcam), test there, then run this to bring
them into kneo-roi-line-alert-xtion/. Files that differ on purpose between the two versions
(listed in VERSION_SPECIFIC) and local files (config.yaml, .env, models, .venv, ...) are never
touched. If you change a VERSION_SPECIFIC file, update the other version by hand.
"""

from __future__ import annotations

import argparse
import filecmp
import fnmatch
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEBCAM = HERE / "kneo-roi-line-alert"
XTION = HERE / "kneo-roi-line-alert-xtion"

# Intentionally different between the two versions.
VERSION_SPECIFIC = {
    "kneo_alert/camera.py",          # webcam: cv2.VideoCapture only / Xtion: + OpenNI2
    "kneo_alert/_openni_compat.py",  # Xtion only
    "deploy/build_openni2.sh",       # Xtion only
    "deploy/check_openni2.py",       # Xtion only
    "tests/test_openni2.py",         # Xtion only
    "tests/test_camera.py",          # webcam only (camera.source: auto)
    "requirements.txt",              # Xtion adds `openni`
    "config.example.yaml",           # camera.source comments differ
    "README.md",
    "TODO.md",
}

# Machine-local / generated files: never copied.
LOCAL_PATTERNS = [
    ".venv/*", "snapshots/*", "*/__pycache__/*", "__pycache__/*", ".pytest_cache/*",
    "models/*", "config.yaml", ".env", "board_config.yaml", "camera_test.jpg", "roi_preview.jpg",
    "*.jpg", "*.jpeg", "*.png", "*.tgz",  # snapshots / pictures copied around by hand
]


def is_local(rel: str) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in LOCAL_PATTERNS)


def shared_files(root: Path) -> set[str]:
    out = set()
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if not is_local(rel) and rel not in VERSION_SPECIFIC:
                out.add(rel)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="actually copy (default: just show)")
    args = ap.parse_args()

    src, dst = shared_files(WEBCAM), shared_files(XTION)
    new = sorted(src - dst)
    changed = sorted(f for f in src & dst if not filecmp.cmp(WEBCAM / f, XTION / f, shallow=False))
    only_xtion = sorted(dst - src)

    if not new and not changed:
        print("Already in sync.")
    for f in new:
        print(f"  + {f}")
    for f in changed:
        print(f"  ~ {f}")
    for f in only_xtion:
        print(f"  ? {f}  (only in Xtion version; not touched — delete it by hand if it was removed on purpose)")

    if args.apply:
        for f in new + changed:
            (XTION / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(WEBCAM / f, XTION / f)
        if new or changed:
            print(f"Copied {len(new) + len(changed)} file(s) to {XTION.name}/")
    elif new or changed:
        print("Run with --apply to copy these.")


if __name__ == "__main__":
    main()
