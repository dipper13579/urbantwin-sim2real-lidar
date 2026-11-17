"""Two-wheeler objects: share the confidence of a bike-shaped object between the classes that label it (label-free rules).

11/10 evidence (validation analysis, real val sets): the real labels file a ridden bicycle either under Bicycle or under Person (a
~2.0 x 0.8 m "Person" box: 4-5 % of the Person labels on M4 val, 17-26 % on M6 val), and the detector cannot tell the two apart
(same two-wheeler score, mean .33-.41).  On such an object the package carries
  - a short Person box (the rider, found by the pedestrian head; mean score .28-.44) that matches neither label, and
  - the long Person copy at a mean score of only .10.

Objects: Bicycle / Motorcycle boxes with score >= --min-score, grouped greedily by centre distance (< --link, best score first);
s_tw = the best two-wheeler score of the object.
  --near T,W     short Person boxes (l < --person-min-l) within --link of an object with s_tw >= T: score x W
  --k-person K   every object gets one long Person box = its best two-wheeler box with score 1 - (1 - s_tw) ** K; long Person
                 boxes within --link of the object that score lower are dropped
  --share W      Bicycle and Motorcycle labels have the same box (2.0 x 0.8 x 1.9) and the detector cannot tell them apart either:
                 the class in which the object scores lower gets the object's best box with score W x s_tw, unless a box of that
                 class within --link already scores at least that

  python tools/tw_object.py --pred X.json --out Y.json --near 0.3,0.3 --k-person 1.5
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import numpy as np

TW = ("Bicycle", "Motorcycle")


def objects(dets, min_score=0.1, link=0.6):
    """Best two-wheeler box of every bike-shaped object of one frame."""
    objs = []
    for d in sorted((d for d in dets if d["class"] in TW and d["score"] >= min_score), key=lambda d: -d["score"]):
        b = d["box"]
        if not any((o["box"][0] - b[0]) ** 2 + (o["box"][1] - b[1]) ** 2 < link ** 2 for o in objs):
            objs.append(d)
    return objs


def share_class(dets, objs, w, link=0.6):
    """Boxes to add: the other two-wheeler class of every object, score w x s_tw, where no box of that class reaches it."""
    add = []
    for o in objs:
        other = TW[1 - TW.index(o["class"])]; b = o["box"]
        best = max((d["score"] for d in dets if d["class"] == other and (d["box"][0] - b[0]) ** 2 + (d["box"][1] - b[1]) ** 2 < link ** 2),
                   default=0.0)
        if best < w * o["score"]:
            add.append(dict(o, **{"class": other, "score": float(w * o["score"])}))
    return add


def apply(preds, near=None, k_person=None, min_score=0.1, link=0.6, person_min_l=1.5, share=None):
    """-> (new predictions, [objects, short Person boxes down-weighted, long Person boxes dropped])."""
    out = {}; n = np.zeros(3, dtype=int)
    for fid, dets in preds.items():
        objs = objects(dets, min_score, link)
        if objs and share:
            dets = list(dets) + share_class(dets, objs, share, link)
        if not objs or not (near or k_person):
            out[fid] = list(dets); continue
        xy = np.array([o["box"][:2] for o in objs]); s = np.array([o["score"] for o in objs]); n[0] += len(objs)
        new = []
        for d in dets:
            if d["class"] == "Person":
                hit = np.hypot(xy[:, 0] - d["box"][0], xy[:, 1] - d["box"][1]) < link
                if hit.any():
                    if d["box"][3] < person_min_l:
                        if near and s[hit].max() >= near[0]:
                            d = dict(d, score=d["score"] * near[1]); n[1] += 1
                    elif k_person and d["score"] <= 1 - (1 - s[hit].max()) ** k_person:
                        n[2] += 1; continue
            new.append(d)
        if k_person:
            new += [dict(o, **{"class": "Person", "score": float(1 - (1 - o["score"]) ** k_person)}) for o in objs]
        out[fid] = new
    return out, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near", default=None, help="T,W: short Person boxes on an object with s_tw >= T get score x W")
    ap.add_argument("--k-person", type=float, default=None, help="long Person box per object with score 1 - (1 - s_tw) ** K")
    ap.add_argument("--min-score", type=float, default=0.1)
    ap.add_argument("--link", type=float, default=0.6)
    ap.add_argument("--person-min-l", type=float, default=1.5)
    ap.add_argument("--share", type=float, default=None, help="W: the other two-wheeler class of an object gets its best box with score W x s_tw")
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text())
    near = tuple(float(v) for v in args.near.split(",")) if args.near else None
    out, n = apply(preds, near, args.k_person, args.min_score, args.link, args.person_min_l, args.share)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    f = max(len(out), 1)
    print(f"{args.out}: {len(out)} frames, per frame: objects {n[0] / f:.1f}, short Person boxes down-weighted {n[1] / f:.1f}, long Person boxes replaced {n[2] / f:.1f}")


if __name__ == "__main__":
    main()
