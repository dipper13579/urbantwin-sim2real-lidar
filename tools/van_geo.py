"""Van geometry re-score (label-free): a van is a vehicle whose roof is clearly above a car's and below a lorry's.

11/10 evidence (validation analysis, real val sets, organizer class naming): 80-90 % of the vehicle-like Van labels are covered at
IoU .5 by a Van box of the package, but the Van head fires on every car (5 Van boxes per frame with score >= .3 against 1-2 Van
labels), so Van is a ranking problem.  The roof height measured in the point cloud separates the two.

For every Van box with score >= --min-score: the run of points above ground + 0.4 m along the box heading through its centre
(lv_geo.measure) gives the top (98th percentile above the local ground) and the run length L.
  score := score x (floor + (1 - floor) x g(top, L))
  g = 0 below ramp[0], rising to 1 at ramp[1], 1 up to 2.7 m, falling to 0.2 at 3.1 m (lorries); x 0.3 for runs of 7.5 m or more
Van boxes below --min-score and boxes without a measurable run get the floor.  Boxes, classes and all other scores are untouched.

  python tools/van_geo.py --pred X.json --frames <dir with frame id .bin> --out Y.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

from scipy.spatial import cKDTree

from bg_prior import load
from lv_geo import measure

FLOOR, MIN_SCORE, RAMP = 0.15, 0.1, (1.8, 2.0)


def gvan(top, L, ramp=RAMP):
    if top < ramp[0]:
        g = 0.0
    elif top < ramp[1]:
        g = (top - ramp[0]) / (ramp[1] - ramp[0])
    elif top <= 2.7:
        g = 1.0
    elif top < 3.1:
        g = 1.0 - 0.8 * (top - 2.7) / 0.4
    else:
        g = 0.2
    return g * (0.3 if L >= 7.5 else 1.0)


def measure_vans(preds, fdir, min_score=MIN_SCORE):
    """{(fid, index in the frame): (top, L) or None} for the Van boxes with score >= min_score."""
    out = {}
    for fid, dets in preds.items():
        sel = [i for i, d in enumerate(dets) if d["class"] == "Van" and d["score"] >= min_score]
        if not sel:
            continue
        pts = load(Path(fdir) / f"{fid}.bin"); tree = cKDTree(pts[:, :2])
        for i in sel:
            b = dets[i]["box"]; r = measure(pts, tree, b[0], b[1], b[6], b[4])
            out[(fid, i)] = (r["top"], r["L"]) if r and r["n"] >= 5 else None
    return out


def rescore(preds, meas, floor=FLOOR, ramp=RAMP):
    out = {}
    for fid, dets in preds.items():
        new = []
        for i, d in enumerate(dets):
            if d["class"] == "Van":
                m = meas.get((fid, i))
                d = dict(d, score=d["score"] * (floor + (1 - floor) * (gvan(m[0], m[1], ramp) if m else 0.0)))
            new.append(d)
        out[fid] = new
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--floor", type=float, default=FLOOR)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--ramp", default=",".join(map(str, RAMP)), help="roof height where the weight starts to rise, and where it reaches 1")
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text())
    meas = measure_vans(preds, args.frames, args.min_score)
    out = rescore(preds, meas, args.floor, tuple(float(v) for v in args.ramp.split(",")))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    n = max(len(out), 1)
    print(f"{args.out}: {len(out)} frames, Van boxes measured per frame {len(meas) / n:.1f}, with a run {sum(v is not None for v in meas.values()) / n:.1f}")


if __name__ == "__main__":
    main()
