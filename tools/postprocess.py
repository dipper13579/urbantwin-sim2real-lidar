"""Class-level post-processing of predictions.json, tuned on the real validation set.

  --relabel "Bus,Truck:10.5"     boxes of either class get Truck if length < 10.5 m, else Bus
  --copies "Bus>Truck:0.8,..."   also emit each Bus box as Truck with score * 0.8 (AP is per class, so a
                                 copy only competes inside its target class)
  --modes "Car=set,Truck=set"    size calibration (see calibrate_sizes.py), applied last

  python tools/postprocess.py --pred in.json --out out.json --relabel "Bus,Truck:10.5" --copies "Bus>Truck:0.8"
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def bev_nms(dets, thr=0.5):
    """Original stable greedy NMS, with same-class 8 m and conservative circle prefilters.

    The official IoU function and comparison threshold are unchanged. Every
    potentially overlapping kept box is tested in original order, including the distance-8 boundary.
    """
    sys.path.insert(0, str(KIT_ROOT / "lumpi" / "scoring_program"))
    from detection_eval import _bev_iou
    keep = []
    groups = {}
    for d in sorted(dets, key=lambda d: -d["score"]):
        b = np.array(d["box"])
        cls = d["class"]
        if cls not in groups:
            groups[cls] = (np.empty((len(dets), 7)), 0)
        boxes, n = groups[cls]
        distance = np.hypot(boxes[:n, 0] - b[0], boxes[:n, 1] - b[1])
        candidates = np.flatnonzero(~(distance > 8))
        if thr > 0 and len(candidates):
            radii = 0.5 * np.hypot(boxes[candidates, 3], boxes[candidates, 4])
            radius = 0.5 * np.hypot(b[3], b[4])
            candidates = candidates[distance[candidates] <= radii + radius + 1e-8]
        if all(_bev_iou(boxes[i], b) < thr for i in candidates):
            keep.append(d)
            boxes[n] = b
            groups[cls] = (boxes, n + 1)
    return keep




def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--relabel", default=None)
    ap.add_argument("--copies", default=None)
    ap.add_argument("--modes", default=None)
    ap.add_argument("--downweight", default=None,
                    help="'Truck:5.0:2.3:0.1' -> Truck boxes shorter than 5.0 m AND lower than 2.3 m get score*0.1")
    ap.add_argument("--dim-scale", default=None, help="'Truck=1,1.14,1.13' scales l,w,h (box bottom kept)")
    ap.add_argument("--dim-set", default=None, help="'pedestrian=0.66,0.69,1.72;...' sets l,w,h (box bottom kept)")
    ap.add_argument("--z-shift", default=None, help="'pedestrian=0.09;...' adds dz to the box center")
    ap.add_argument("--radial-shift", default=None,
                    help="'vehicle=-0.19;...' moves the box center by dr along the ray from the sensor (origin)")
    ap.add_argument("--max-per-frame", type=int, default=250)
    args = ap.parse_args()
    preds = json.loads(args.pred.read_text())

    for fid, dets in preds.items():
        if args.relabel:
            pair, thr = args.relabel.split(":")
            small, large = pair.split(",")[1], pair.split(",")[0]   # "Bus,Truck:10.5" -> Truck below, Bus above
            for d in dets:
                if d["class"] in (small, large):
                    d["class"] = small if d["box"][3] < float(thr) else large
        if args.copies:
            extra = []
            for rule in args.copies.split(","):
                src_dst, w = rule.split(":")
                src, dst = src_dst.split(">")
                extra += [{"box": list(d["box"]), "score": round(d["score"] * float(w), 5), "class": dst}
                          for d in dets if d["class"] == src]
            dets = dets + extra
        if args.relabel or args.copies:
            dets = bev_nms(dets)
        if args.downweight:
            cls, min_l, min_h, w = args.downweight.split(":")
            for d in dets:
                if d["class"] == cls and d["box"][3] < float(min_l) and d["box"][5] < float(min_h):
                    d["score"] = round(d["score"] * float(w), 5)
        if args.dim_scale:
            for rule in args.dim_scale.split(";"):
                cls, s = rule.split("=")
                s = [float(v) for v in s.split(",")]
                for d in dets:
                    if d["class"] == cls:
                        h0 = d["box"][5]
                        d["box"][3:6] = [round(d["box"][3 + i] * s[i], 4) for i in range(3)]
                        d["box"][2] += (d["box"][5] - h0) / 2
        if args.dim_set:
            for rule in args.dim_set.split(";"):
                cls, s = rule.split("=")
                s = [float(v) for v in s.split(",")]
                for d in dets:
                    if d["class"] == cls:
                        h0 = d["box"][5]
                        d["box"][3:6] = s
                        d["box"][2] += (s[2] - h0) / 2
        if args.z_shift:
            for rule in args.z_shift.split(";"):
                cls, dz = rule.split("=")
                for d in dets:
                    if d["class"] == cls:
                        d["box"][2] = round(d["box"][2] + float(dz), 4)
        if args.radial_shift:
            for rule in args.radial_shift.split(";"):
                cls, dr = rule.split("=")
                for d in dets:
                    if d["class"] == cls:
                        r = max(np.hypot(d["box"][0], d["box"][1]), 1e-6)
                        f = max(r + float(dr), 0.0) / r
                        d["box"][0], d["box"][1] = round(d["box"][0] * f, 4), round(d["box"][1] * f, 4)
        preds[fid] = sorted(dets, key=lambda d: -d["score"])[: args.max_per_frame]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    if args.modes:
        subprocess.run([sys.executable, str(ROOT / "tools" / "calibrate_sizes.py"), "--pred", str(args.out),
                        "--out", str(args.out), "--modes", args.modes], check=True, stdout=subprocess.DEVNULL)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
