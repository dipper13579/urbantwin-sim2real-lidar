"""Two-wheeler box refit: heading / centre from the point cluster under the box (label-free) + label-statistics footprint.

11/10 evidence (validation analysis, validation analysis, real val sets): two-wheeler-sized objects (Bicycle, Motorcycle and the
~2.0 x 0.8 m "Person" labels: 4 % of the Person labels on M4 val, 17-26 % on M6 val, 33 % inside the test window) are found - the box
centre is within 0.1 m - but 30-60 % of them miss IoU .5: the detector heading is off by 16-27 deg (median; the main axis of the
point cluster by 7-15 deg) and the footprint by 0.1-0.2 m.

For every box of --classes and every Person box at least --person-min-l long:
  footprint  l, w := --size (median of the two-wheeler labels of the non-forbidden real frames: box-size statistics, the use the
             organizers allow; Bicycle, Motorcycle and long Person agree within 0.05 m on every set)
  heading    := main axis of the cluster (single linkage 0.30 m of the points 0.25-2.3 m above the box bottom, the component with
             most points within 0.5 m of the box centre) when the cluster is elongated (1.0-2.6 m along, >= 1.4 x across);
             the centre moves to the middle of the cluster extent.  Otherwise heading and centre stay.
Height, bottom, class and score are untouched; boxes of other classes pass through.

  python tools/tw_refit.py --pred X.json --frames <dir with frame id .bin> --out Y.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from bg_prior import load

SIZE = (1.95, 0.86)
LINK, R, NEAR, MIN_PTS = 0.30, 1.9, 0.5, 8
ALONG, RATIO = (1.0, 2.6), 1.4


def cluster_fit(pts, tree, b, zs=(0.25, 2.3), pca_top=None):
    """(heading of the main axis, centre xy, extent along, extent across) of the cluster under box b, or None."""
    idx = tree.query_ball_point([b[0], b[1]], R)
    if len(idx) < MIN_PTS:
        return None
    p = pts[idx]
    bot = b[2] - b[5] / 2
    p = p[(p[:, 2] > bot + zs[0]) & (p[:, 2] < bot + zs[1])]
    if len(p) < MIN_PTS:
        return None
    close = np.hypot(p[:, 0] - b[0], p[:, 1] - b[1]) < NEAR
    if not close.any():
        return None
    pairs = cKDTree(p[:, :2]).query_pairs(LINK, output_type="ndarray")
    n = len(p)
    lab = connected_components(coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)), directed=False)[1] if len(pairs) else np.arange(n)
    c = p[lab == np.bincount(lab[close]).argmax()]
    if len(c) < MIN_PTS:
        return None
    q = c[c[:, 2] < bot + pca_top] if pca_top else c
    if len(q) < MIN_PTS:
        q = c
    ax = np.linalg.eigh(np.cov((q[:, :2] - q[:, :2].mean(0)).T))[1][:, 1]
    nrm = np.array([-ax[1], ax[0]]); m = c[:, :2].mean(0); xy = c[:, :2] - m
    la, ha = np.percentile(xy @ ax, [2, 98]); lo, ho = np.percentile(xy @ nrm, [2, 98])
    return float(np.arctan2(ax[1], ax[0])), m + ax * (la + ha) / 2 + nrm * (lo + ho) / 2, float(ha - la), float(ho - lo)


def elongated(f):
    return f is not None and ALONG[0] <= f[2] <= ALONG[1] and f[2] >= RATIO * f[3]


def refit_box(b, f, size=SIZE, yaw=True, resize=True):
    b = list(b)
    if yaw and elongated(f):
        d = (f[0] - b[6] + np.pi / 2) % np.pi - np.pi / 2          # the axis has no direction: stay on the detector's side
        b[0], b[1], b[6] = float(f[1][0]), float(f[1][1]), float(b[6] + d)
    if resize:
        b[3], b[4] = size
    return b


def selected(d, classes, person_min_l):
    return d["class"] in classes or (d["class"] == "Person" and d["box"][3] >= person_min_l)


def refit(preds, fdir, classes=("Bicycle", "Motorcycle"), person_min_l=1.5, size=SIZE, yaw=True, resize=True, zs=(0.25, 2.3), pca_top=None,
          fits=None):
    """-> (new predictions, number of treated boxes, number with an elongated cluster).  fits: optional cache {(fid, x, y, bottom): fit}."""
    out = {}; n = nel = 0
    for fid, dets in preds.items():
        sel = [i for i, d in enumerate(dets) if selected(d, classes, person_min_l)]
        new = list(dets)
        if sel:
            pts = tree = None
            for i in sel:
                b = dets[i]["box"]; key = (fid, round(b[0], 2), round(b[1], 2), round(b[2] - b[5] / 2, 1))
                if fits is None or key not in fits:
                    if yaw and pts is None:
                        pts = load(Path(fdir) / f"{fid}.bin"); tree = cKDTree(pts[:, :2])
                    f = cluster_fit(pts, tree, b, zs, pca_top) if yaw else None
                    if fits is not None:
                        fits[key] = f
                else:
                    f = fits[key]
                new[i] = dict(dets[i], box=refit_box(b, f, size, yaw, resize)); n += 1; nel += elongated(f)
        out[fid] = new
    return out, n, nel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--classes", default="Bicycle,Motorcycle")
    ap.add_argument("--person-min-l", type=float, default=1.5, help="Person boxes at least this long are treated as two-wheeler-sized")
    ap.add_argument("--size", default=",".join(map(str, SIZE)), help="footprint l,w")
    ap.add_argument("--no-yaw", action="store_true")
    ap.add_argument("--no-size", action="store_true")
    ap.add_argument("--pca-top", type=float, default=None, help="main axis from the cluster points below this height above the box bottom")
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text())
    out, n, nel = refit(preds, args.frames, tuple(args.classes.split(",")), args.person_min_l, tuple(float(v) for v in args.size.split(",")),
                        not args.no_yaw, not args.no_size, pca_top=args.pca_top)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{args.out}: {len(out)} frames, two-wheeler-sized boxes per frame {n / max(len(out), 1):.1f}, with an elongated cluster {nel / max(n, 1):.2f}")


if __name__ == "__main__":
    main()
