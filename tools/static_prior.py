"""Static false-positive prior for a fixed roadside site: poles, bushes and sign posts get detected as Person /
Bicycle in almost every frame, real road users rarely stay on the same spot. The map is the share of frames, over
unlabeled real frames from OTHER time windows (non-forbidden, >= 300 frames from any forbidden id), in which a
prediction of that class sits within the 3x3 neighbourhood of a BEV cell. Predictions of the target set that land
on a recurring cell (share >= --q) get their score multiplied by --w.

  python tools/static_prior.py --pred target.json --out out.json --map-preds a.json b.json --classes Person,Bicycle \
      --q 0.2 --w 0.3
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from collections import defaultdict

import numpy as np


def cell_of(x, y, size):
    return int(np.floor(x / size)), int(np.floor(y / size))


def build(map_preds, classes, size, min_score):
    frames = defaultdict(lambda: defaultdict(set))  # cls -> cell -> {frame keys}
    n = 0
    for k, path in enumerate(map_preds):
        preds = json.load(open(path))
        for fid, dets in preds.items():
            n += 1
            for d in dets:
                if d["class"] in classes and d["score"] >= min_score:
                    i, j = cell_of(d["box"][0], d["box"][1], size)
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            frames[d["class"]][(i + di, j + dj)].add((k, fid))
    return {c: {cell: len(fs) / n for cell, fs in cells.items()} for c, cells in frames.items()}, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map-preds", nargs="+", required=True)
    ap.add_argument("--classes", default="Person,Bicycle")
    ap.add_argument("--q", type=float, default=0.2, help="share of map frames above which a cell counts as static")
    ap.add_argument("--w", type=float, default=0.3, help="score multiplier on static cells")
    ap.add_argument("--cell", type=float, default=0.5)
    ap.add_argument("--min-score", type=float, default=0.3, help="map: only predictions above this score count")
    args = ap.parse_args()
    classes = set(args.classes.split(","))
    share, n = build(args.map_preds, classes, args.cell, args.min_score)
    preds = json.load(open(args.pred))
    hit = defaultdict(int); tot = defaultdict(int)
    for dets in preds.values():
        for d in dets:
            if d["class"] not in classes:
                continue
            tot[d["class"]] += 1
            if share.get(d["class"], {}).get(cell_of(d["box"][0], d["box"][1], args.cell), 0.0) >= args.q:
                d["score"] *= args.w
                hit[d["class"]] += 1
        dets.sort(key=lambda d: -d["score"])
    json.dump(preds, open(args.out, "w"))
    print(f"map frames {n}; down-weighted " + ", ".join(f"{c} {hit[c]}/{tot[c]}" for c in sorted(tot)))


if __name__ == "__main__":
    main()
