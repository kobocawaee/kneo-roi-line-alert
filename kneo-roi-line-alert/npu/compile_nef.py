"""Quantize + compile the cut YOLOv8 (person) and UltraFace (face) models into ONE KL730 NEF.

Runs INSIDE the Kneron toolchain Docker image. Use kneron/toolchain:v0.25.0: the KNEO Pi KL730
firmware (2.0.0) only runs multi-model NEFs from v0.25/v0.26; with v0.33.1 the second model hangs
the NPU (and can reboot the board). The conda env must be on PATH (the compiler calls `python`):

    docker run --rm -v <work>:/data kneron/toolchain:v0.25.0 bash -c \
        'export PATH=/workspace/miniconda/envs/onnx1.13/bin:$PATH; python /data/compile_nef.py --work /data'

<work> is either the PROJECT folder itself (simplest; run from it in PowerShell):

    docker run --rm -v "${PWD}:/data" -w /data kneron/toolchain:v0.25.0 bash -c \
        'export PATH=/workspace/miniconda/envs/onnx1.13/bin:$PATH; python npu/compile_nef.py'

  using npu/yolov8n_320_cut.onnx, npu/ultraface_320_cut.onnx, kneo_alert/*_post.py, calib/*.jpg
  -> npu/build/nef/models_730.nef

or a flat folder holding yolov8n_320_cut.onnx, ultraface_320_cut.onnx, yolo_post.py,
face_post.py and calib/*.jpg  -> <work>/nef/models_730.nef.
Model ids in the NEF: 32769 = person, 32770 = face. A float-vs-quantized score comparison
is printed at the end.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import onnx
from PIL import Image

import ktc

PLATFORM = "730"
PERSON_ID, FACE_ID = 32769, 32770
YOLO_CUT = FACE_CUT = ""  # set in main()
YOLO_IN, FACE_IN = "images", "input"


# ---- preprocessing: MUST match what the board feeds the NPU (kneo_alert/npu_detector.py) ----
def letterbox(img: Image.Image, size: int = 320) -> tuple[np.ndarray, float, int, int]:
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (left, top))
    return np.asarray(canvas), r, left, top


def yolo_input(path: str) -> np.ndarray:
    rgb, *_ = letterbox(Image.open(path).convert("RGB"))
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]          # 1x3x320x320


def face_input(path: str) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB").resize((320, 240), Image.BILINEAR))
    return ((rgb.astype(np.float32) - 127.0) / 128.0).transpose(2, 0, 1)[None]  # 1x3x240x320


def optimize(path: str, dst: str) -> tuple[str, str]:
    """Kneron-optimize an ONNX and save it to its OWN file (ModelConfig(onnx_model=...) writes every
    model to the same temp path, so two in-memory models overwrite each other). Returns (path, input name)."""
    m = onnx.load(path)
    if m.opset_import[0].version < 11:
        from onnx import version_converter
        m = version_converter.convert_version(m, 11)
    m = ktc.onnx_optimizer.onnx2onnx_flow(m, eliminate_tail=False)
    onnx.save(m, dst)
    return dst, m.graph.input[0].name


def pick(*paths: str) -> str:
    for p in paths:
        if os.path.isfile(p):
            return p
    sys.exit(f"Not found: {paths[0]} (also looked in {', '.join(paths[1:])}). Run npu/cut_*.py first.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.getcwd())
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-calib", type=int, default=100)
    args = ap.parse_args()
    work = os.path.abspath(args.work)
    project = os.path.isdir(os.path.join(work, "kneo_alert"))
    sys.path.insert(0, work)
    sys.path.insert(0, os.path.join(work, "kneo_alert"))   # yolo_post / face_post for the check
    root = os.path.join(work, "npu", "build") if project else work
    out = os.path.join(root, "out")
    os.makedirs(out, exist_ok=True)
    global YOLO_CUT, FACE_CUT
    YOLO_CUT = pick(os.path.join(work, "npu", "yolov8n_320_cut.onnx"), os.path.join(work, "yolov8n_320_cut.onnx"))
    FACE_CUT = pick(os.path.join(work, "npu", "ultraface_320_cut.onnx"), os.path.join(work, "ultraface_320_cut.onnx"))

    calib = sorted(glob.glob(os.path.join(work, "calib", "*.jpg")))[: args.max_calib]
    if len(calib) < 10:
        sys.exit(f"Need at least 10 calibration images in {work}/calib (found {len(calib)})")
    print(f"[calib] {len(calib)} images")

    person_onnx, person_in = optimize(YOLO_CUT, os.path.join(out, "person_opt.onnx"))
    face_onnx, face_in = optimize(FACE_CUT, os.path.join(out, "face_opt.onnx"))
    print(f"[optimize] person input={person_in!r}  face input={face_in!r}")
    km_person = ktc.ModelConfig(PERSON_ID, "0001", PLATFORM, onnx_path=person_onnx)
    km_face = ktc.ModelConfig(FACE_ID, "0001", PLATFORM, onnx_path=face_onnx)
    for name, km in (("person", km_person), ("face", km_face)):
        rep = str(km.evaluate())
        print(f"[evaluate] {name}: " + " | ".join(l.strip() for l in rep.splitlines()
                                                  if "fps" in l.lower() or "cpu_node" in l or "Success" in l))

    print("[analysis] person (quantization)...")
    person_bie = km_person.analysis({person_in: [yolo_input(p) for p in calib]}, output_dir=os.path.join(out, "person"),
                       threads=args.threads)
    print("[analysis] face (quantization)...")
    face_bie = km_face.analysis({face_in: [face_input(p) for p in calib]}, output_dir=os.path.join(out, "face"),
                     threads=args.threads)

    # The toolchain may clean its output folders later (compile / simulator), so keep copies.
    bie_dir = os.path.join(root, "bie")
    os.makedirs(bie_dir, exist_ok=True)
    person_bie = shutil.copy(person_bie, os.path.join(bie_dir, "person.bie"))
    face_bie = shutil.copy(face_bie, os.path.join(bie_dir, "face.bie"))

    print("[compile] both models into one NEF...")
    nef = ktc.compile([km_person, km_face])
    # Keep the result in its own folder: the E2E simulator unpacks NEFs next to the file
    # and can wipe that folder if it crashes.
    nef_dir = os.path.join(root, "nef")
    os.makedirs(nef_dir, exist_ok=True)
    dst = os.path.join(nef_dir, "models_730.nef")
    shutil.copy(nef, dst)
    print(f"[compile] NEF: {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")

    sim_dir = "/tmp/nefcheck"
    os.makedirs(sim_dir, exist_ok=True)
    sim_nef = os.path.join(sim_dir, "models_730.nef")
    shutil.copy(dst, sim_nef)
    check(work, sim_nef, calib[5::20], person_in, face_in,
          {PERSON_ID: person_bie, FACE_ID: face_bie})


def check(work: str, nef: str, images: list[str], person_in: str, face_in: str, bies: dict) -> None:
    """Compare float ONNX vs the quantized model, after our CPU post-processing.
    Uses the NEF hardware simulator; falls back to the BIE fixed-point simulator if that fails."""
    mode = {"kind": "nef"}

    def npu(x, model_id, in_name):
        if mode["kind"] == "nef":
            try:
                return ktc.kneron_inference([x], nef_file=nef, model_id=model_id, platform=int(PLATFORM),
                                            input_names=[in_name])
            except Exception as e:  # noqa: BLE001
                print(f"  (NEF simulator failed: {e}; using BIE fixed-point simulator instead)")
                mode["kind"] = "bie"
        return ktc.kneron_inference([x], bie_file=bies[model_id], platform=int(PLATFORM), input_names=[in_name])
    from face_post import decode_faces
    from yolo_post import decode_person

    # Output order differs between simulators; head shapes are unique, so order by shape.
    yolo_order = [(64, 40, 40), (80, 40, 40), (64, 20, 20), (80, 20, 20), (64, 10, 10), (80, 10, 10)]
    face_order = [(6, 30, 40), (12, 30, 40), (4, 15, 20), (8, 15, 20), (4, 8, 10), (8, 8, 10),
                  (6, 4, 5), (12, 4, 5)]

    def by_shape(outs, order):
        d = {tuple(np.asarray(o).shape[-3:]): np.asarray(o).reshape(np.asarray(o).shape[-3:]) for o in outs}
        return [d[s] for s in order]

    def yolo_heads(outs):
        o = by_shape(outs, yolo_order)
        return [(o[2 * i], o[2 * i + 1]) for i in range(3)]

    def face_heads(outs):
        o = by_shape(outs, face_order)
        return [(o[2 * k], o[2 * k + 1]) for k in range(4)]

    # Float reference: onnxruntime when available (the toolchain's float simulator gave wrong
    # person scores in our tests), else the toolchain's own float inference.
    try:
        import onnxruntime as ort
        sess = {YOLO_CUT: ort.InferenceSession(YOLO_CUT), FACE_CUT: ort.InferenceSession(FACE_CUT)}

        def floatrun(path, x, name):
            return sess[path].run(None, {sess[path].get_inputs()[0].name: x.astype(np.float32)})
    except ImportError:
        def floatrun(path, x, name):
            return ktc.kneron_inference([x], onnx_file=path, input_names=[name])

    print("\n[check] float ONNX vs quantized KL730 model (person / face max score per image)")
    for p in images:
        yi, fi = yolo_input(p), face_input(p)
        yf = floatrun(YOLO_CUT, yi, YOLO_IN)
        yn = npu(yi, PERSON_ID, person_in)
        ff = floatrun(FACE_CUT, fi, FACE_IN)
        fn = npu(fi, FACE_ID, face_in)
        if p == images[0]:
            print(f"  [{mode['kind']}] person output shapes:", [np.asarray(o).shape for o in yn])
            print(f"  [{mode['kind']}] face   output shapes:", [np.asarray(o).shape for o in fn])
        pb_f, ps_f = decode_person(yolo_heads(yf), conf_threshold=0.45)
        pb_n, ps_n = decode_person(yolo_heads(yn), conf_threshold=0.45)
        fb_f, fs_f = decode_faces(face_heads(ff), 0.7)
        fb_n, fs_n = decode_faces(face_heads(fn), 0.7)
        print(f"  {os.path.basename(p):32s} person max score float={ps_f.max(initial=0):.2f} npu={ps_n.max(initial=0):.2f}"
              f" | face max score float={fs_f.max(initial=0):.2f} npu={fs_n.max(initial=0):.2f}")


if __name__ == "__main__":
    main()
