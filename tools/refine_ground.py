"""Per-object ground height: put each box bottom at the local ground (from the frame's own points) + a per-class
offset, instead of one constant z shift for the whole scene. The ground is a low percentile of the point
heights in an annulus around the box (inner radius = box half-diagonal + --inner, outer = that + --outer),
within --zwin below / 0.5 m above the current box bottom, so the object's own points are left out.

  new_z = (1 - alpha) * z + alpha * (ground + offset + h / 2)

Calibration (no output written): --calib-gt gt.json prints, per class, the median and MAD of
(box bottom - local ground) for the GT boxes and for the predictions (score >= --min-score).

  python tools/refine_ground.py --pred in.json --frames data/lumpi_real/points_M4 --calib-gt data/lumpi_real/gt_M4.json \
      --classes Person,Car
  python tools/refine_ground.py --pred in.json --frames DIR --out out.json --offset "Person=0.02;Car=0.05"
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


def ground_under(b, p, tree, args):
    """Local ground height around box b (7-DoF), or None when too few ground points."""
    r0 = 0.5 * np.hypot(b[3], b[4]) + args.inner
    idx = tree.query_ball_point(b[:2], r0 + args.outer)
    if not idx:
        return None
    q = p[idx]
    q = q[np.hypot(q[:, 0] - b[0], q[:, 1] - b[1]) > r0]
    bottom = b[2] - b[5] / 2
    q = q[(q[:, 2] > bottom - args.zwin) & (q[:, 2] < bottom + 0.5)]
    if len(q) < args.min_pts:
        return None
    return float(np.percentile(q[:, 2], args.pct))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--classes", default="Person")
    ap.add_argument("--offset", default="", help="'cls=dz;...' box bottom above the local ground (default 0)")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--inner", type=float, default=0.3)
    ap.add_argument("--outer", type=float, default=1.5)
    ap.add_argument("--zwin", type=float, default=1.0)
    ap.add_argument("--pct", type=float, default=30)
    ap.add_argument("--min-pts", type=int, default=10)
    ap.add_argument("--calib-gt", type=Path)
    ap.add_argument("--min-score", type=float, default=0.3)
    args = ap.parse_args()
    classes = args.classes.split(",")
    offset = {k: float(v) for k, v in (r.split("=") for r in args.offset.split(";") if r)}
    preds = json.loads(args.pred.read_text())
    gt = json.loads(args.calib_gt.read_text()) if args.calib_gt else None
    stats = {(src, c): [] for src in ("gt", "pred") for c in classes}
    moved = total = 0
    for fid, dets in preds.items():
        sets = [("pred", [d for d in dets if d["class"] in classes])]
        if gt is not None:
            sets.append(("gt", [g for g in gt.get(fid, []) if g["class"] in classes]))
        if not any(s for _, s in sets):
            continue
        p = load(args.frames / f"{fid}.bin").astype(np.float64)
        tree = cKDTree(p[:, :2])
        for src, objs in sets:
            for d in objs:
                b = d["box"]
                if gt is not None and src == "pred" and d.get("score", 1) < args.min_score:
                    continue
                g = ground_under(np.array(b, dtype=np.float64), p, tree, args)
                if gt is not None:
                    if g is not None:
                        stats[(src, d["class"])].append(b[2] - b[5] / 2 - g)
                    continue
                total += 1
                if g is None:
                    continue
                moved += 1
                target = g + offset.get(d["class"], 0.0) + b[5] / 2
                b[2] = round(float((1 - args.alpha) * b[2] + args.alpha * target), 4)
    if gt is not None:
        for c in classes:
            line = f"{c:<11}"
            for src in ("gt", "pred"):
                a = np.array(stats[(src, c)])
                if len(a):
                    line += f"  {src}: n={len(a):5d} median={np.median(a):+.3f} MAD={np.median(np.abs(a - np.median(a))):.3f}"
            print(line)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"wrote {args.out}: moved {moved}/{total} boxes of {args.classes}")


if __name__ == "__main__":
    main()
