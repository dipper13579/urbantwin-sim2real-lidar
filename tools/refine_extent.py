"""Point-based length refinement of large-vehicle boxes (Bus / Truck): measure the frame's points along each
box's heading and fit the box length (and center) to their extent.

On real LUMPI M4 the detectors predict ~11.2 m for 8.45 m trucks (they fire the Bus head, trained on 12 m
buses), so length-based Bus/Truck relabeling sends most trucks to Bus. Run this before postprocess.py.

Per box: points within the box's width (+ --w-margin) and height (above bottom + --z-ground, below top +
0.3 m), and within half-length + --l-margin along the heading; extent = percentile(--pct, 100 - --pct) of
their along-heading coordinate + --pad. Shrink-only by default (--grow allows up to the searched margin).
new length = (1 - alpha) * old + alpha * extent; the center moves to the middle of the measured extent.

  python tools/refine_extent.py --pred raw.json --frames data/lumpi_real/points_M4 --out out.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load(path):
    raw = np.fromfile(str(path), dtype=np.float32)
    return raw.reshape(-1, 3) if raw.size % 3 == 0 else raw.reshape(-1, 4)[:, :3]  # official loader order


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--classes", default="Bus,Truck")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l-margin", type=float, default=0.5)
    ap.add_argument("--w-margin", type=float, default=0.2)
    ap.add_argument("--z-ground", type=float, default=0.3)
    ap.add_argument("--pct", type=float, default=1.0)
    ap.add_argument("--pad", type=float, default=0.2)
    ap.add_argument("--min-pts", type=int, default=30)
    ap.add_argument("--min-len", type=float, default=4.0)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--grow", action="store_true")
    args = ap.parse_args()
    classes = set(args.classes.split(","))
    preds = json.loads(args.pred.read_text())
    changed = total = 0
    dl = []
    for fid, dets in preds.items():
        sel = [d for d in dets if d["class"] in classes and d["score"] >= args.min_score]
        if not sel:
            continue
        p = load(args.frames / f"{fid}.bin").astype(np.float64)
        tree = cKDTree(p[:, :2])
        for d in sel:
            total += 1
            x, y, z, l, w, h, yaw = d["box"]
            c = np.array([x, y])
            hv = np.array([np.cos(yaw), np.sin(yaw)])
            nv = np.array([-hv[1], hv[0]])
            q = p[tree.query_ball_point(c, np.hypot(l / 2 + args.l_margin, w / 2 + args.w_margin))]
            rel = q[:, :2] - c
            a, b = rel @ hv, rel @ nv
            bottom = z - h / 2
            m = ((np.abs(b) <= w / 2 + args.w_margin) & (np.abs(a) <= l / 2 + args.l_margin)
                 & (q[:, 2] > bottom + args.z_ground) & (q[:, 2] < z + h / 2 + 0.3))
            if m.sum() < args.min_pts:
                continue
            lo, hi = np.percentile(a[m], [args.pct, 100 - args.pct])
            ext = max(hi - lo + args.pad, args.min_len)
            if ext >= l and not args.grow:
                continue
            new_l = (1 - args.alpha) * l + args.alpha * ext
            shift = args.alpha * (lo + hi) / 2
            d["box"][0], d["box"][1] = round(float(x + shift * hv[0]), 4), round(float(y + shift * hv[1]), 4)
            d["box"][3] = round(float(new_l), 4)
            dl.append(new_l - l)
            changed += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"wrote {args.out}: refined {changed}/{total} {args.classes} boxes, median length change "
          f"{np.median(dl) if dl else 0:+.2f} m")


if __name__ == "__main__":
    main()
