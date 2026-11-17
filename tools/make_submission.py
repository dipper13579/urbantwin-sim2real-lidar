"""Pack a Codabench submission for the UrbanTwin Sim2Real challenge, then run the
official local_eval.py on it as a format check.

submission.zip
  synthetic/0001.bin ... 0050.bin   float32 N x 3 (xyz only), <= 1.5 MB (125,000 points) each
  predictions.json                  {frame_id: [{"box": [x,y,z,dx,dy,dz,heading], "score": s, "class": c}, ...]}
  declaration.txt                   honor statement + name + date

Example:
  python tools/make_submission.py --track lumpi \
      --synthetic-dir outputs/lumpi/synthetic \
      --predictions outputs/lumpi/predictions.json \
      --declaration outputs/declaration.txt \
      --out outputs/lumpi/submission.zip

`--synthetic-dir` may hold more than 50 .bin/.npy frames: the first 50 (sorted) are
masked to the track range, uniformly subsampled to the point cap, and renumbered.
Write frames as N x 3 here: the official loader tries N x 3 before N x 4, so an
N x 4 file whose float count divides by 3 would be misread.
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = 50
MAX_BYTES = 1_500_000
MAX_POINTS = MAX_BYTES // 12
DECLARATION_PHRASE = "I confirm that I did not train on any frame listed in"


def load_xyz(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.asarray(np.load(path)[:, :3], dtype=np.float32)
    raw = np.fromfile(path, dtype=np.float32)
    return (raw.reshape(-1, 3) if raw.size % 3 == 0 else raw.reshape(-1, 4)[:, :3]).copy()


def fit_frame(pts: np.ndarray, pc_range, rng) -> np.ndarray:
    lo, hi = np.asarray(pc_range[:3]), np.asarray(pc_range[3:])
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    if len(pts) > MAX_POINTS:
        pts = pts[np.sort(rng.choice(len(pts), size=MAX_POINTS, replace=False))]
    return np.ascontiguousarray(pts, dtype=np.float32)


def check_predictions(preds: dict, frame_ids, classes) -> list:
    errors = []
    allowed = {c.lower() for c in classes}
    if set(preds) != set(frame_ids):
        errors.append(f"frame ids differ: missing={sorted(set(frame_ids) - set(preds))[:3]} "
                      f"extra={sorted(set(preds) - set(frame_ids))[:3]}")
    for fid, dets in preds.items():
        for i, d in enumerate(dets):
            box = d.get("box")
            if not (isinstance(box, list) and len(box) == 7 and all(math.isfinite(float(v)) for v in box)):
                errors.append(f"{fid}[{i}]: box must be 7 finite floats")
            elif min(box[3:6]) <= 0:
                errors.append(f"{fid}[{i}]: non-positive box size {box[3:6]}")
            if not 0.0 <= float(d.get("score", -1)) <= 1.0:
                errors.append(f"{fid}[{i}]: score must be in [0, 1]")
            if str(d.get("class", "")).lower() not in allowed:
                errors.append(f"{fid}[{i}]: class {d.get('class')!r} not in {classes}")
    return errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", choices=["lumpi", "v2x"], required=True)
    ap.add_argument("--synthetic-dir", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--declaration", type=Path, required=True, help="declaration.txt (or a signed .pdf)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-check", action="store_true", help="skip the official local_eval.py format check")
    args = ap.parse_args()

    kit = KIT_ROOT / args.track
    config = json.loads((kit / "reference_data_dummy" / "config.json").read_text())
    frame_ids = json.loads((kit / "detection_test_frame_ids.json").read_text())
    preds = json.loads(args.predictions.read_text())

    errors = check_predictions(preds, frame_ids, config["class_names"])
    if args.declaration.suffix == ".txt" and DECLARATION_PHRASE.lower() not in args.declaration.read_text().lower():
        errors.append("declaration.txt does not contain the required statement")
    files = sorted(p for p in args.synthetic_dir.iterdir() if p.suffix in (".bin", ".npy"))
    if len(files) < N_FRAMES:
        errors.append(f"need {N_FRAMES} synthetic frames, found {len(files)}")
    if errors:
        raise SystemExit("Submission NOT written:\n  - " + "\n  - ".join(errors[:30]))

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, f in enumerate(files[:N_FRAMES], start=1):
            zf.writestr(f"synthetic/{i:04d}.bin", fit_frame(load_xyz(f), config["point_cloud_range"], rng).tobytes())
        zf.writestr("predictions.json", json.dumps(preds))
        zf.write(args.declaration, f"declaration{args.declaration.suffix}")
    n_dets = sum(len(v) for v in preds.values())
    print(f"wrote {args.out}  ({len(preds)} frames, {n_dets} detections)")

    if not args.no_check:
        format_check(kit, args.out.resolve())


def format_check(kit: Path, zip_path: Path):
    """Run the official local_eval.py as a pure format check.

    The kit's dummy reference holds only 5 frames (n_realism_frames=5), which a real 50-frame
    submission fails, so build a 50-frame stand-in from the submission's own synthetic frames.
    The scores it prints are meaningless (empty GT, self-comparison); only validity matters.
    Never put detection-test or other forbidden frames in a reference set.
    """
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / "ref"
        shutil.copytree(kit / "reference_data_dummy", ref)
        cfg = json.loads((ref / "config.json").read_text())
        cfg["n_realism_frames"] = N_FRAMES
        (ref / "config.json").write_text(json.dumps(cfg))
        shutil.rmtree(ref / "realism_reference")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.startswith("synthetic/") and name.endswith(".bin"):
                    zf.extract(name, ref)
        (ref / "synthetic").rename(ref / "realism_reference")
        subprocess.run([sys.executable, str(kit / "local_eval.py"), str(zip_path), str(ref)], cwd=kit, check=True)
    scores = json.loads((kit / "scores.json").read_text())
    if not scores.get("valid_submission"):
        raise SystemExit(f"INVALID submission: {scores.get('error_message')}")
    print("format check passed (valid_submission=1)")


if __name__ == "__main__":
    main()
