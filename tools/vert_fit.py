"""Point-fitted vertical placement of boxes (label-free, uses only the frame's own points). On V2X real val the
GT box bottoms are noisy against the local ground (pedestrian MAD 8 cm, vehicle 14 cm) but a perfect z is worth
+0.06 mAP: annotators fit boxes to the visible points. Per box, zmin / zmax = 2nd / 98th percentile of the points
inside the (shrunk) BEV footprint within [bottom-0.6, top+0.6]; the box is moved to
    top:    centre = zmax - h/2        bottom: centre = zmin + h/2
    mid:    centre = (zmin + zmax)/2   fit:    h = zmax - zmin + margin, centre = midpoint
blended with the current box by alpha. Settings chosen on the two real-val halves (validation analysis).

  python tools/vert_fit.py --pred in.json --frames DIR --out out.json --spec "pedestrian=mid:0.5;vehicle=top:1.0"
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bg_prior import load

SHRINK = {"vehicle": 0.85, "truck": 0.85, "Car": 0.85, "Truck": 0.85, "Van": 0.85, "Bus": 0.85}
MINPTS = {"vehicle": 20, "truck": 20, "Car": 20, "Truck": 20, "Van": 20, "Bus": 20}


def zext(pts, tree, box, cls):
    x, y, z, dx, dy, dz, h = box; s = SHRINK.get(cls, 1.0)
    hl, hw = max(dx * s / 2, 0.35), max(dy * s / 2, 0.35)
    p = pts[tree.query_ball_point([x, y], float(np.hypot(hl, hw)))]
    if len(p) < MINPTS.get(cls, 5):
        return None
    c, sn = np.cos(h), np.sin(h)
    u = (p[:, 0] - x) * c + (p[:, 1] - y) * sn; v = -(p[:, 0] - x) * sn + (p[:, 1] - y) * c
    m = (np.abs(u) <= hl) & (np.abs(v) <= hw) & (p[:, 2] > z - dz / 2 - 0.6) & (p[:, 2] < z + dz / 2 + 0.6)
    if m.sum() < MINPTS.get(cls, 5):
        return None
    q = np.percentile(p[m, 2], [2, 98])
    return float(q[0]), float(q[1])


def place(box, ext, mode, alpha, margin=0.1):
    b = list(box); zmin, zmax = ext; hgt = b[5]
    if mode == "fit":
        hgt = min(max(zmax - zmin + margin, 0.5 * b[5]), 1.5 * b[5]); zc = (zmin + zmax) / 2
    elif mode == "top":
        zc = zmax - hgt / 2
    elif mode == "bottom":
        zc = zmin + hgt / 2
    else:
        zc = (zmin + zmax) / 2
    b[2] = (1 - alpha) * b[2] + alpha * zc; b[5] = (1 - alpha) * b[5] + alpha * hgt
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True, help="directory of <frame_id>.bin")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", required=True, help="'cls=mode:alpha;...' with mode in top, bottom, mid, fit")
    args = ap.parse_args()
    spec = {}
    for item in args.spec.split(";"):
        cls, rest = item.split("="); mode, alpha = rest.split(":"); spec[cls] = (mode, float(alpha))
    preds = json.loads(args.pred.read_text()); out = {}; n = {c: [0, 0] for c in spec}
    for k, dets in preds.items():
        pts = load(args.frames / f"{k}.bin"); tree = cKDTree(pts[:, :2]); nd = []
        for d in dets:
            d = dict(d)
            if d["class"] in spec:
                n[d["class"]][1] += 1; e = zext(pts, tree, d["box"], d["class"])
                if e is not None:
                    d["box"] = place(d["box"], e, *spec[d["class"]]); n[d["class"]][0] += 1
            nd.append(d)
        out[k] = nd
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print("vert_fit: " + ", ".join(f"{c} {a}/{b} boxes ({spec[c][0]} a{spec[c][1]})" for c, (a, b) in n.items()))


if __name__ == "__main__":
    main()
