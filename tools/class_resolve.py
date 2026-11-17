"""Two-wheeler vs pedestrian resolution for LUMPI. Real LUMPI labels a cyclist as one Bicycle box (rider included,
~2.0 x 0.85 x 1.9 m, no separate Person), and the detector puts many Bicycle / Motorcycle boxes on pedestrians. The
above-ground points' extent along the box heading separates the two: a pedestrian is short along any heading, a
bicycle is long along its own. Bicycle / Motorcycle boxes with extent < --lo get score x --w. No labels are used
here; lo and w were chosen on the four real val sets (validation analysis).

Tried and dropped (val): down-weighting Person boxes next to a long Bicycle box (features/apply with hi < 99) hurt
Person on every set, and the heading-free principal-axis extent (pca_extent) separated worse than the heading one.

  python tools/class_resolve.py --pred outputs/lumpi/OGPSB/predictions.json \
      --frames starting_kit/lumpi/detection_test_frames --out outputs/lumpi/OGPSBC/predictions.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bg_prior import load


def extent(pts, tree, box, half_len=1.4, half_w=0.45, lo=0.2, hi=2.2, min_pts=5):
    """5-95 percentile extent of the above-ground points along the box heading, inside a 2.8 x 0.9 m strip;
    None when fewer than min_pts points."""
    x, y, z, dx, dy, dz, h = box
    idx = tree.query_ball_point([x, y], float(np.hypot(half_len, half_w)))
    if len(idx) < min_pts:
        return None
    p = pts[idx]; bottom = z - dz / 2
    p = p[(p[:, 2] > bottom + lo) & (p[:, 2] < bottom + hi)]
    c, s = np.cos(h), np.sin(h)
    u = (p[:, 0] - x) * c + (p[:, 1] - y) * s
    v = -(p[:, 0] - x) * s + (p[:, 1] - y) * c
    m = (np.abs(u) <= half_len) & (np.abs(v) <= half_w)
    if m.sum() < min_pts:
        return None
    q = np.percentile(u[m], [5, 95])
    return float(q[1] - q[0])


def pca_extent(pts, tree, box, radius=1.2, lo=0.2, hi=2.2, min_pts=5):
    """Heading-free variant: 5-95 percentile extent along the principal axis of the above-ground points within
    radius of the box centre."""
    x, y, z, dx, dy, dz, h = box
    idx = tree.query_ball_point([x, y], radius)
    if len(idx) < min_pts:
        return None
    p = pts[idx]; bottom = z - dz / 2
    p = p[(p[:, 2] > bottom + lo) & (p[:, 2] < bottom + hi), :2]
    if len(p) < min_pts:
        return None
    p = p - p.mean(0)
    u = p @ np.linalg.eigh(p.T @ p)[1][:, -1]
    q = np.percentile(u, [5, 95])
    return float(q[1] - q[0])


def features(pts, dets, r):
    """Per box: Bicycle -> its extent; Person -> the largest extent among Bicycle boxes centred within r (else None)."""
    tree = cKDTree(pts[:, :2])
    ext = [extent(pts, tree, d["box"]) if d["class"] == "Bicycle" else None for d in dets]
    bikes = [(d["box"][0], d["box"][1], e) for d, e in zip(dets, ext) if e is not None]
    out = []
    for d, e in zip(dets, ext):
        if d["class"] == "Person":
            near = [b[2] for b in bikes if np.hypot(d["box"][0] - b[0], d["box"][1] - b[1]) < r]
            out.append(max(near) if near else None)
        else:
            out.append(e)
    return out


def apply(dets, feat, lo, wb, hi, wp):
    out = []
    for d, f in zip(dets, feat):
        d = dict(d)
        if f is not None and d["class"] == "Bicycle" and f < lo:
            d["score"] *= wb
        if f is not None and d["class"] == "Person" and f >= hi:
            d["score"] *= wp
        out.append(d)
    return sorted(out, key=lambda d: -d["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True, help="directory of <frame_id>.bin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", default="Bicycle,Motorcycle")
    ap.add_argument("--lo", type=float, default=0.7)
    ap.add_argument("--w", type=float, default=0.5)
    args = ap.parse_args()
    classes = set(args.classes.split(","))
    preds = json.loads(args.pred.read_text()); out = {}; n = tot = 0
    for k, dets in preds.items():
        pts = load(args.frames / f"{k}.bin"); tree = cKDTree(pts[:, :2]); nd = []
        for d in dets:
            d = dict(d)
            if d["class"] in classes:
                tot += 1; e = extent(pts, tree, d["box"])
                if e is not None and e < args.lo:
                    d["score"] *= args.w; n += 1
            nd.append(d)
        out[k] = sorted(nd, key=lambda d: -d["score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"down-weighted {n}/{tot} {args.classes} boxes (extent < {args.lo} m, x {args.w})")


if __name__ == "__main__":
    main()
