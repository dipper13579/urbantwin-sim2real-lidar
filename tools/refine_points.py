"""Point-based refinement of small boxes (pedestrians): pull each box center toward the median of the frame's
points around it, pushed back along the sensor ray by --radial-offset (the LiDAR only sees the near
surface, so the point median sits ~6 cm in front of the true center on V2X-Real).

  new_xy = (1 - alpha) * pred_xy + alpha * (median_xy(points within --radius) + radial_offset * ray)

  python tools/refine_points.py --pred in.json --frames data/v2x_real/points --out out.json \
      --cls pedestrian --alpha 0.5
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
    ap.add_argument("--cls", default="pedestrian")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--radius", type=float, default=0.4)
    ap.add_argument("--radial-offset", type=float, default=0.063)
    ap.add_argument("--min-pts", type=int, default=5)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--sensors", default=None,
                    help="'x,y;x,y;...' sensor positions; the ray for --radial-offset starts at the nearest one "
                         "(default: a single sensor at the origin)")
    args = ap.parse_args()
    sensors = (np.array([[float(v) for v in s.split(",")[:2]] for s in args.sensors.split(";")])
               if args.sensors else np.zeros((1, 2)))
    preds = json.loads(args.pred.read_text())
    moved = total = 0
    for fid, dets in preds.items():
        sel = [d for d in dets if d["class"] == args.cls]
        if not sel:
            continue
        p = load(args.frames / f"{fid}.bin").astype(np.float64)
        tree = cKDTree(p[:, :2])
        for d in sel:
            total += 1
            b = d["box"]
            c = np.array(b[:2], dtype=np.float64)
            zlo, zhi = b[2] - b[5] / 2 - 0.1, b[2] + b[5] / 2 + 0.1
            ok = False
            for _ in range(args.iters):
                near = p[tree.query_ball_point(c, args.radius)]
                near = near[(near[:, 2] > zlo) & (near[:, 2] < zhi)]
                if len(near) < args.min_pts:
                    break
                m = np.median(near[:, :2], 0)
                v = m - sensors[np.argmin(np.hypot(*(sensors - m).T))]
                u = v / max(np.linalg.norm(v), 1e-6)
                c = (1 - args.alpha) * c + args.alpha * (m + args.radial_offset * u)
                ok = True
            if ok:
                moved += 1
                b[0], b[1] = round(float(c[0]), 4), round(float(c[1]), 4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"wrote {args.out}: refined {moved}/{total} {args.cls} boxes")


if __name__ == "__main__":
    main()
