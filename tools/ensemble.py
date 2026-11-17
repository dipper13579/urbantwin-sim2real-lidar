"""Weighted box fusion of several predictions.json files (same frames).

Per frame and class, boxes are clustered greedily by BEV IoU (official IoU code); each cluster becomes
one box whose x, y, z, l, w, h are the score-weighted mean, heading is taken from the top-scoring
member, and score = sum(scores) / n_models (so boxes only one model found are down-weighted).

  python tools/ensemble.py --preds a.json b.json --out ens.json [--iou 0.4]
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KIT_ROOT / "lumpi" / "scoring_program"))
from detection_eval import _bev_iou  # noqa: E402  official


def fuse(dets, n_models, iou_thr):
    clusters = []
    # vectorized prefilter over cluster refs: boxes farther apart than the sum of their half-diagonals cannot
    # overlap, so only those candidates (in cluster order, as before) get the exact IoU test
    ref_xy = np.zeros((len(dets), 2))
    ref_r = np.zeros(len(dets))
    for d in sorted(dets, key=lambda d: -d["score"]):
        b = np.array(d["box"])
        r = 0.5 * np.hypot(b[3], b[4])
        n = len(clusters)
        dist = np.hypot(*(ref_xy[:n] - b[:2]).T)
        for i in np.flatnonzero((dist < 6) & (dist < ref_r[:n] + r)):
            if _bev_iou(clusters[i]["ref"], b) >= iou_thr:
                clusters[i]["members"].append(d)
                break
        else:
            ref_xy[n], ref_r[n] = b[:2], r
            clusters.append({"ref": b, "members": [d]})
    out = []
    for c in clusters:
        s = np.array([m["score"] for m in c["members"]])
        B = np.array([m["box"] for m in c["members"]])
        box = list((B[:, :6] * s[:, None]).sum(0) / s.sum()) + [float(B[0, 6])]
        out.append({"box": [round(float(v), 4) for v in box], "score": round(float(min(1.0, s.sum() / n_models)), 5),
                    "class": c["members"][0]["class"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--iou", type=float, default=0.4)
    ap.add_argument("--max-per-frame", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1, help="frames are independent: fuse them in a process pool (same output)")
    args = ap.parse_args()
    global _RUNS, _ARGS
    _RUNS = [json.loads(p.read_text()) for p in args.preds]
    _ARGS = args
    fids = list(_RUNS[0])
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:  # fork: workers share the loaded runs
            res = pool.map(_fuse_frame, fids, chunksize=4)
    else:
        res = [_fuse_frame(f) for f in fids]
    out = dict(zip(fids, res))
    runs = _RUNS
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}: {len(out)} frames, {sum(map(len, out.values()))} boxes from {len(runs)} runs")


_RUNS, _ARGS = None, None


def _fuse_frame(fid):
    dets = [d for r in _RUNS for d in r.get(fid, [])]
    fused = []
    for cls in sorted({d["class"] for d in dets}):
        fused += fuse([d for d in dets if d["class"] == cls], len(_RUNS), _ARGS.iou)
    return sorted(fused, key=lambda d: -d["score"])[: _ARGS.max_per_frame]


if __name__ == "__main__":
    main()
