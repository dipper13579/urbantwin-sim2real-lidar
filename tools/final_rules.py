"""Final label-free rules on a stacked LUMPI package (outputs/lumpi/win/prereg.md 2c).  Every frame is processed on its own, from
its own points and boxes; nothing refers to a position or to an individual object.

  --truck-van W   every Truck box shorter than --max-l is added as a Van box with score W x s, unless a Van box with its centre
                  within --rad already scores at least that (the chain copies the other way with Van>Truck:0.2)
  --sparse W      Person / Car boxes with fewer than --min-pts points of the frame inside the box: score x W

  python tools/final_rules.py --pred X.json --frames <dir with frame id .bin> --truck-van 0.5 --sparse 0.3 --out Y.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bg_prior import load

MAX_L = 7.5     # a van is shorter (label sizes up to 6.0 m); the "long" size group starts here
RAD = 1.0
MIN_PTS = 20
SPARSE_CLASSES = ("Person", "Car")


def truck_van(preds, w, max_l=MAX_L, rad=RAD):
    """-> (new predictions, number of Van boxes added)."""
    out = {}; n = 0
    for fid, dets in preds.items():
        vans = [d for d in dets if d["class"] == "Van"]
        vxy = np.array([d["box"][:2] for d in vans], dtype=float).reshape(-1, 2); vs = np.array([d["score"] for d in vans], dtype=float)
        add = []
        for d in dets:
            if d["class"] != "Truck" or d["box"][3] >= max_l:
                continue
            s = float(w * d["score"]); b = d["box"]
            if len(vs) and (vs[np.hypot(vxy[:, 0] - b[0], vxy[:, 1] - b[1]) < rad] >= s).any():
                continue
            add.append(dict(d, **{"class": "Van", "score": s}))
        out[fid] = list(dets) + add; n += len(add)
    return out, n


def count_inside(pts, tree, boxes):
    """Number of points inside each oriented box (rows x, y, z, l, w, h, yaw); tree = cKDTree(pts[:, :2])."""
    B = np.asarray(boxes, dtype=float).reshape(-1, 7); n = np.zeros(len(B), dtype=int)
    idx = tree.query_ball_point(B[:, :2], 0.5 * np.hypot(B[:, 3], B[:, 4]) + 1e-6)
    for i, (b, ix) in enumerate(zip(B, idx)):
        if ix:
            p = pts[ix]; c, s = np.cos(b[6]), np.sin(b[6]); ex, ey = p[:, 0] - b[0], p[:, 1] - b[1]
            n[i] = int(np.sum((np.abs(ex * c + ey * s) <= b[3] / 2) & (np.abs(-ex * s + ey * c) <= b[4] / 2) & (np.abs(p[:, 2] - b[2]) <= b[5] / 2)))
    return n


def sparse(preds, frames, w, classes=SPARSE_CLASSES, min_pts=MIN_PTS):
    """-> (new predictions, Counter of down-weighted boxes per class)."""
    out = {}; n = Counter()
    for fid, dets in preds.items():
        sel = [i for i, d in enumerate(dets) if d["class"] in classes]; new = list(dets)
        if sel:
            pts = load(Path(frames) / f"{fid}.bin").astype(np.float64)
            cnt = count_inside(pts, cKDTree(pts[:, :2]), [dets[i]["box"] for i in sel])
            for i, c in zip(sel, cnt):
                if c < min_pts:
                    new[i] = dict(dets[i], score=float(dets[i]["score"] * w)); n[dets[i]["class"]] += 1
        out[fid] = new
    return out, n


def apply(preds, frames, r):
    """r: recipe with the optional keys "truck_van": W and "sparse": {"w": W, "classes": [...]}."""
    if r.get("truck_van"):
        preds, _ = truck_van(preds, r["truck_van"])
    if r.get("sparse"):
        preds, _ = sparse(preds, frames, r["sparse"]["w"], tuple(r["sparse"].get("classes", SPARSE_CLASSES)))
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--truck-van", type=float, default=None)
    ap.add_argument("--max-l", type=float, default=MAX_L)
    ap.add_argument("--rad", type=float, default=RAD)
    ap.add_argument("--sparse", type=float, default=None)
    ap.add_argument("--sparse-classes", default=",".join(SPARSE_CLASSES))
    ap.add_argument("--min-pts", type=int, default=MIN_PTS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text()); msg = []
    if args.truck_van:
        preds, n = truck_van(preds, args.truck_van, args.max_l, args.rad); msg.append(f"Van boxes added {n}")
    if args.sparse:
        preds, n = sparse(preds, args.frames, args.sparse, tuple(args.sparse_classes.split(",")), args.min_pts); msg.append(f"sparse boxes down-weighted {dict(n)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(preds))
    print(f"{args.out}: {len(preds)} frames, {sum(len(v) for v in preds.values())} boxes; " + "; ".join(msg))


if __name__ == "__main__":
    main()
