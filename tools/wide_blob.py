"""Wide tall objects for LUMPI (Truck): a box for objects that are wider than a road vehicle's body (working machines, vehicles with
an extended arm or load, articulated vehicles in a turn). Works on a finished prediction file; every frame is processed on its own,
from its own points and boxes only.

The detector and tools/lv_geo.py limit a box to the width of a vehicle body (<= 3.67 m), so an object whose labelled box is wider
than that can never reach IoU 0.5. For every Truck / Bus box with score >= MIN_SCORE:

 1. blob      the connected set of points 0.5-4.8 m above the local ground (grid CELL, one dilation, 8-neighbours) that is nearest
              to the box centre (<= SEED_R).
 2. test      the minimum-area rectangle of the blob's points >= TALL above the ground (neighbouring cars and people that touch the
              blob stay below it) is at least W_WIDE wide and at most L_MAX long, and the blob does not reach the border of the
              2 R window (buildings, vegetation).
 3. box       that rectangle, ground to the blob's point top, class Truck, ADDED with score = --alpha x the best Truck / Bus score
              among the boxes seeded on the blob. The seed boxes stay (--keep K multiplies their scores by K, 0 removes them).
              Defaults keep 1.0, alpha 0.5: replacing the seeds, or alpha 1.0, lost on the older validation sets
              (outputs/lumpi/ev1110_wide1.txt, ev1110_wide2.txt: the blob of a real lorry often merges with a neighbour).

W_WIDE, TALL and the grid were set on the points under labelled boxes of real validation frames (non-forbidden frames of the test
window, validation analysis: ordinary lorries, box trucks, vans, cars and buses are not called wide); no position and no
individual object enters the rule.

  python tools/wide_blob.py --pred outputs/lumpi/XL/predictions.json --frames starting_kit/lumpi/detection_test_frames \
      --out outputs/lumpi/XLW/predictions.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from bg_prior import load

LV = ("Truck", "Bus")
MIN_SCORE = 0.05
CELL, R, SEED_R = 0.25, 12.0, 2.5
TALL, W_WIDE, L_MAX = 2.0, 4.0, 20.0


def rect(b, lo=1, hi=99):
    """Minimum-area rectangle of the points b (percentile extents, heading searched in 2 degree steps): x, y, l, w, yaw."""
    best = None
    for a in np.radians(np.arange(0, 90, 2)):
        c, s = np.cos(a), np.sin(a); u = b[:, 0] * c + b[:, 1] * s; v = -b[:, 0] * s + b[:, 1] * c
        u0, u1 = np.percentile(u, [lo, hi]); v0, v1 = np.percentile(v, [lo, hi])
        if best is None or (u1 - u0) * (v1 - v0) < best[0]:
            best = ((u1 - u0) * (v1 - v0), a, u0, u1, v0, v1)
    _, a, u0, u1, v0, v1 = best; c, s = np.cos(a), np.sin(a); cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    l, w, yaw = u1 - u0, v1 - v0, a
    if w > l:
        l, w, yaw = w, l, a + np.pi / 2
    return float(cu * c - cv * s), float(cu * s + cv * c), float(l), float(w), float(yaw)


def wide_fit(pts, tree, x, y, w_wide=W_WIDE, tall=TALL, l_max=L_MAX, h_max=None):
    """Box of the wide tall blob under (x, y), or None."""
    p = pts[tree.query_ball_point([x, y], R * 1.42)]
    p = p[(np.abs(p[:, 0] - x) < R) & (np.abs(p[:, 1] - y) < R)]
    if len(p) < 30:
        return None
    near = p[np.hypot(p[:, 0] - x, p[:, 1] - y) < 4.0]
    g = float(np.percentile(near[:, 2] if len(near) >= 20 else p[:, 2], 3))
    q = p[(p[:, 2] > g + 0.5) & (p[:, 2] < g + 4.8)]
    if len(q) < 30:
        return None
    n = int(2 * R / CELL); ix = np.clip(((q[:, 0] - x + R) / CELL).astype(int), 0, n - 1); iy = np.clip(((q[:, 1] - y + R) / CELL).astype(int), 0, n - 1)
    occ = np.zeros((n, n), bool); occ[ix, iy] = True
    lab, _ = ndimage.label(ndimage.binary_dilation(occ, iterations=1), structure=np.ones((3, 3)))
    ids = lab[ix, iy]; d = np.hypot(q[:, 0] - x, q[:, 1] - y)
    if d.min() > SEED_R:
        return None
    m = ids == ids[np.argmin(d)]
    if m.sum() < 30 or ix[m].min() == 0 or iy[m].min() == 0 or ix[m].max() == n - 1 or iy[m].max() == n - 1:
        return None
    b = q[m]; t = b[b[:, 2] > g + tall]
    if len(t) < 20:
        return None
    bx, by, l, w, yaw = rect(t)
    if w < w_wide or l > l_max:
        return None
    top = float(np.percentile(b[:, 2], 99))
    if h_max and top - g > h_max:
        return None
    return [bx, by, (g + top) / 2, l, w, top - g, yaw]


def inside(box, x, y):
    c, s = np.cos(box[6]), np.sin(box[6]); ex, ey = x - box[0], y - box[1]
    return abs(ex * c + ey * s) <= box[3] / 2 and abs(-ex * s + ey * c) <= box[4] / 2


def apply(preds, fdir, keep=0.0, w_wide=W_WIDE, tall=TALL, min_score=MIN_SCORE, alpha=1.0, l_max=L_MAX, h_max=None):
    """-> (new predictions, [wide boxes, seed boxes])."""
    out, n = {}, [0, 0]
    for fid in sorted(preds):
        dets = preds[fid]; seeds = [d for d in dets if d["class"] in LV and d["score"] >= min_score]
        if not seeds:
            out[fid] = dets; continue
        pts = load(Path(fdir) / f"{fid}.bin"); tree = cKDTree(pts[:, :2]); wide, used = [], set()
        for d in sorted(seeds, key=lambda d: -d["score"]):
            hit = next((w for w in wide if inside(w["box"], d["box"][0], d["box"][1])), None)
            if hit is None:
                box = wide_fit(pts, tree, d["box"][0], d["box"][1], w_wide, tall, l_max, h_max)
                if box is None:
                    continue
                if not any(inside(w["box"], box[0], box[1]) or inside(box, w["box"][0], w["box"][1]) for w in wide):
                    wide.append({"box": [float(v) for v in box], "score": float(d["score"] * alpha), "class": "Truck"})
            used.add(id(d))
        res = [dict(d, score=float(d["score"] * keep)) if id(d) in used else d for d in dets if keep > 0 or id(d) not in used]
        out[fid] = res + wide; n[0] += len(wide); n[1] += len(used)
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=1.0, help="weight of the seed boxes (0 = removed)")
    ap.add_argument("--alpha", type=float, default=0.5, help="score of the wide box = alpha x best seed score")
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text())
    out, n = apply(preds, args.frames, args.keep, alpha=args.alpha)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{args.out}: {len(out)} frames, wide boxes per frame {n[0] / max(len(out), 1):.2f}, seed boxes per frame {n[1] / max(len(out), 1):.2f}")


if __name__ == "__main__":
    main()
