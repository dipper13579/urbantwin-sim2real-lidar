"""Label-free point-feature score rules for pedestrian boxes (V2X). Two thirds of the detector's pedestrian boxes
on the real site sit on unlabelled low, sparse clusters (median 5 body points, 0.86 m tall) while true pedestrians
have ~16 body points and reach 1.5 m. Per box: n_body = points within 0.45 m of the centre between bottom+0.3 and
top+0.1, top = highest such point above the bottom, n_norm = n_body * (range/20)^2. Scores are multiplied by --wb
when n_body (or n_norm with --norm) < --nb and by --wt when top < --top. Thresholds chosen on the two real-val
halves (validation analysis).

  python tools/ped_rules.py --pred in.json --frames DIR --out out.json --nb 6 --wb 0.3 --top 1.2 --wt 0.5
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from bg_prior import load


def features(pts, tree, box):
    x, y, z, dx, dy, dz, _ = box; bottom = z - dz / 2
    p = pts[tree.query_ball_point([x, y], 0.45)]
    body = p[(p[:, 2] > bottom + 0.3) & (p[:, 2] < bottom + dz + 0.1)]
    top = (body[:, 2].max() - bottom) if len(body) else 0.0
    return len(body), top, len(body) * (np.hypot(x, y) / 20.0) ** 2


def box_count(pts, tree, box, margin=0.1):
    """Points inside the (heading-aligned) box footprint grown by `margin`, between bottom+0.2 and top+0.2:
    a size-agnostic point-count feature for the larger classes (Car / Truck / Bus / Van), where the 0.45 m body core
    of `features` is meaningless."""
    x, y, z, dx, dy, dz, h = box; bottom = z - dz / 2
    p = pts[tree.query_ball_point([x, y], 0.5 * np.hypot(dx, dy) + margin + 0.1)]
    if not len(p):
        return 0
    c, s = np.cos(h), np.sin(h); ex, ey = p[:, 0] - x, p[:, 1] - y
    lx, ly = ex * c + ey * s, -ex * s + ey * c
    m = (np.abs(lx) < dx / 2 + margin) & (np.abs(ly) < dy / 2 + margin) & (p[:, 2] > bottom + 0.2) & (p[:, 2] < bottom + dz + 0.2)
    return int(m.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cls", default="pedestrian")
    ap.add_argument("--nb", type=float, default=0, help="body-point threshold (0 = off)")
    ap.add_argument("--wb", type=float, default=1.0)
    ap.add_argument("--norm", action="store_true", help="compare range-adjusted counts (n * (range/20)^2) instead of raw counts")
    ap.add_argument("--top", type=float, default=0, help="top-height threshold in m (0 = off)")
    ap.add_argument("--wt", type=float, default=1.0)
    ap.add_argument("--nbox", type=float, default=0, help="in-box point-count threshold (whole footprint; 0 = off)")
    ap.add_argument("--wbox", type=float, default=1.0)
    args = ap.parse_args()
    preds = json.loads(args.pred.read_text()); out = {}; nb = nt = nx = tot = 0
    for k, dets in preds.items():
        pts = load(args.frames / f"{k}.bin"); tree = cKDTree(pts[:, :2]); nd = []
        for d in dets:
            d = dict(d)
            if d["class"] == args.cls:
                tot += 1; n_body, top, n_norm = features(pts, tree, d["box"])
                if args.nb and (n_norm if args.norm else n_body) < args.nb:
                    d["score"] *= args.wb; nb += 1
                if args.top and top < args.top:
                    d["score"] *= args.wt; nt += 1
                if args.nbox:
                    n_box = box_count(pts, tree, d["box"])
                    if args.norm:
                        n_box *= (np.hypot(d["box"][0], d["box"][1]) / 20.0) ** 2
                    if n_box < args.nbox:
                        d["score"] *= args.wbox; nx += 1
            nd.append(d)
        out[k] = sorted(nd, key=lambda d: -d["score"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"ped_rules: {tot} {args.cls} boxes; {nb} below {'n_norm' if args.norm else 'n_body'} {args.nb} (x {args.wb}), "
          f"{nt} below top {args.top} m (x {args.wt}), {nx} below n_box {args.nbox} (x {args.wbox})")


if __name__ == "__main__":
    main()
