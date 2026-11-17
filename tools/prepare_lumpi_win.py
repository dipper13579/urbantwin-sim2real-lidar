"""In-window real LUMPI validation set M4w (frames fetched by setup/132_fetch_win.sh) and organizer-named GT for the older sets.

Organizer-confirmed rule (11/10): every real frame except the 100 forbidden IDs may be used for validation, model selection and
box-size statistics.  M4w = 100 frames of Measurement 4 inside the test window, each >= 3 frames from every forbidden ID.

Points: same processing as prepare_lumpi_rare.py (crop [-40,40]x[-40,40]x[-5,5], 50k-point subsample) -> points_M4w/M4w_{frame}.bin.
GT, written the way the test GT is built (11/10 findings): class names by ORG_CLASS_NAMES (class_id 6 = Van) and *all* boxes of the
frame, not cropped to the point range ("crop": box centre inside the range, "n_pts": points of the cropped cloud inside, "id": track).
  gt_M4w.json              all boxes            (--min-gt-pts 0 = the test convention; --min-gt-pts 1 = visible boxes only)
  gt_M{4,4r,6,6r}o.json    the older GT files with the organizers' naming (Unknown <-> Van), otherwise unchanged

  python tools/prepare_lumpi_win.py
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lumpi_real import ORG_CLASS_NAMES, is_excluded, load_labels, to_org  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(str(RAW_ROOT / 'LUMPI-real'))
OUT = WORK_ROOT / "data" / "lumpi_real"
RANGE = np.array([-40, -40, -5, 40, 40, 5], dtype=np.float64)
N_POINTS = 50_000
SAFE = 2          # frames within this distance of a forbidden ID are never read (labels) and must not be in the folder (points)


def main():
    import torch
    from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu

    m, tag = 4, "M4w"
    (OUT / f"points_{tag}").mkdir(parents=True, exist_ok=True)
    files = sorted((RAW / tag / "lidar").glob("*.ply"))
    frames = np.array([int(f.stem) for f in files])
    assert not is_excluded(frames, SAFE).any(), "a frame next to a forbidden ID slipped in"
    labels = load_labels(OUT / f"M{m}_Label.csv", buffer=SAFE)
    by_frame = {k: g for k, g in labels.groupby("frame")}
    gt = {}
    for f, fr in zip(files, frames):
        fid = f"{tag}_{fr:06d}"
        dst = OUT / f"points_{tag}" / f"{fid}.bin"
        if dst.exists():
            pts = np.fromfile(dst, np.float32).reshape(-1, 3)
        else:
            v = PlyData.read(str(f))["vertex"]
            pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
            pts = pts[np.all((pts >= RANGE[:3]) & (pts <= RANGE[3:]), axis=1)]
            rng = np.random.default_rng(m * 1_000_000 + int(fr))
            if len(pts) > N_POINTS:
                pts = pts[rng.choice(len(pts), N_POINTS, replace=False)]
            pts = pts.astype(np.float32)
            pts.tofile(dst)
        g = by_frame.get(int(fr))
        boxes = []
        if g is not None:
            arr = g[["x", "y", "z", "l", "w", "h", "heading"]].to_numpy(np.float32)
            n_in = points_in_boxes_cpu(torch.from_numpy(pts), torch.from_numpy(arr)).numpy().sum(1) if len(arr) else []
            crop = ((g.x.abs() <= 40) & (g.y.abs() <= 40) & (g.z >= -5) & (g.z <= 5)).to_numpy()
            for b, c, n, k, t in zip(arr, g["cls"].astype(int), n_in, crop, g["id"].astype(int)):
                boxes.append({"box": [float(x) for x in b], "class": ORG_CLASS_NAMES.get(int(c), f"cls{c}"), "n_pts": int(n),
                              "crop": bool(k), "id": int(t)})
        gt[fid] = boxes
    (OUT / f"gt_{tag}.json").write_text(json.dumps(gt))
    allb = [b for v in gt.values() for b in v]
    print(f"{tag}: {len(gt)} frames {frames.min()}..{frames.max()}")
    for name, sel in (("all boxes", allb), ("centre in the crop", [b for b in allb if b["crop"]]), (">= 1 point", [b for b in allb if b["n_pts"] >= 1]),
                      (">= 5 points", [b for b in allb if b["n_pts"] >= 5])):
        print(f"  {name:<19}" + ", ".join(f"{c}={sum(b['class'] == c for b in sel)}" for c in ORG_CLASS_NAMES.values()))

    for s in ("4", "4r", "6", "6r"):
        old = json.loads((OUT / f"gt_M{s}.json").read_text())
        new = to_org(old)
        (OUT / f"gt_M{s}o.json").write_text(json.dumps(new))
        vis = [b for v in new.values() for b in v if b.get("n_pts", 1) >= 1]
        print(f"M{s}o: {len(new)} frames; boxes with >= 1 pt: " + ", ".join(f"{c}={sum(b['class'] == c for b in vis)}" for c in ORG_CLASS_NAMES.values()))


if __name__ == "__main__":
    main()
