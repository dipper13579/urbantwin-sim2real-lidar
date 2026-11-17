"""Class-level box sizes for LUMPI from box-size statistics of real validation labels (non-forbidden frames of the test window,
outputs/lumpi/win/sizes.txt, sizes2.txt). Works on a finished prediction file; only box dimensions change, never a position, a
heading, a score or a class, and every rule is the same for all boxes of a class.

  --person-short L,W   Person boxes shorter than 1.5 m (pedestrians): footprint := L x W
  --car-scale SL,SW    Car: l x SL, w x SW (4.44 / 4.48 and 1.91 / 2.00 move the size mode of the chain to the label median)
  --bicycle-h H        Bicycle: height := H, bottom kept
  --van-w RULE         Van: V1 w := max(w, 1.8); V2 w := 1.8 if l < 4.3 else 2.1; V4 w := max(w, 1.9)
  --truck-w S          Truck: w x S
  --bus-w S            Bus: w x S

  python tools/box_sizes.py --pred X.json --out Y.json --van-w V2 --bicycle-h 1.86
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

PERSON_MAX_L = 1.5
VAN_RULES = ("V1", "V2", "V4")


def van_w(b, rule):
    if rule == "V1":
        return max(b[4], 1.8)
    if rule == "V2":
        return 1.8 if b[3] < 4.3 else 2.1
    if rule == "V4":
        return max(b[4], 1.9)
    raise ValueError(rule)


def resize(b, cls, person_short=None, car_scale=None, bicycle_h=None, van=None, truck_w=None, bus_w=None):
    """New box (list) of one detection, or the same object if no rule applies."""
    if cls == "Person" and person_short and b[3] < PERSON_MAX_L:
        b = list(b); b[3], b[4] = person_short
    elif cls == "Car" and car_scale:
        b = list(b); b[3] *= car_scale[0]; b[4] *= car_scale[1]
    elif cls == "Bicycle" and bicycle_h:
        b = list(b); bottom = b[2] - b[5] / 2; b[5] = bicycle_h; b[2] = bottom + bicycle_h / 2
    elif cls == "Van" and van:
        b = list(b); b[4] = van_w(b, van)
    elif cls == "Truck" and truck_w:
        b = list(b); b[4] *= truck_w
    elif cls == "Bus" and bus_w:
        b = list(b); b[4] *= bus_w
    return b


def apply(preds, **rules):
    """-> (new predictions, {class: number of resized boxes})."""
    out, n = {}, {}
    for fid, dets in preds.items():
        new = []
        for d in dets:
            b = resize(d["box"], d["class"], **rules)
            if b is not d["box"]:
                d = dict(d, box=[float(v) for v in b]); n[d["class"]] = n.get(d["class"], 0) + 1
            new.append(d)
        out[fid] = new
    return out, n


def pair(s):
    return tuple(float(v) for v in s.split(",")) if s else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--person-short", default=None)
    ap.add_argument("--car-scale", default=None)
    ap.add_argument("--bicycle-h", type=float, default=None)
    ap.add_argument("--van-w", default=None, choices=VAN_RULES)
    ap.add_argument("--truck-w", type=float, default=None)
    ap.add_argument("--bus-w", type=float, default=None)
    args = ap.parse_args()
    preds = json.loads(Path(args.pred).read_text())
    out, n = apply(preds, person_short=pair(args.person_short), car_scale=pair(args.car_scale), bicycle_h=args.bicycle_h, van=args.van_w,
                   truck_w=args.truck_w, bus_w=args.bus_w)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"{args.out}: {len(out)} frames, resized boxes " + (", ".join(f"{c} {v}" for c, v in sorted(n.items())) or "none"))


if __name__ == "__main__":
    main()
