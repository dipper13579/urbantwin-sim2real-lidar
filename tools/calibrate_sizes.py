"""Post-hoc box-size calibration of predictions toward real LUMPI annotation conventions.

UT-LUMPI boxes are systematically smaller than LUMPI's (e.g. Bicycle 1.3x0.45x1.3 m vs 2.0x0.86x1.9 m
including the rider). For each class, predicted (l, w, h) are moved toward the real median:
    mode ratio : dims * real_median / ut_median
    mode set   : dims = real_median
    mode blend : geometric mix of the two with --alpha (1 = set, 0 = ratio)
The box bottom stays put (z shifts by the height change / 2).

Real medians come from non-forbidden frames only (tools/lumpi_real.py), UT medians from data/lumpi.

  python tools/calibrate_sizes.py --pred in.json --out out.json --mode ratio [--classes Car,Bicycle]
  python tools/calibrate_sizes.py --print-table
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def size_tables(measurement=4):
    from lumpi_real import CLASS_NAMES, load_labels
    df = load_labels(WORK_ROOT / "data" / "lumpi_real" / f"M{measurement}_Label.csv")
    df = df[(df.x.abs() <= 40) & (df.y.abs() <= 40)]
    real = {CLASS_NAMES[int(c)]: g[["l", "w", "h"]].median().to_numpy() for c, g in df.groupby("cls") if len(g) >= 50}
    infos = pickle.loads((WORK_ROOT / "data" / "lumpi" / "infos_train.pkl").read_bytes())
    boxes, names = np.concatenate([i["gt_boxes"] for i in infos]), np.concatenate([i["gt_names"] for i in infos])
    ut = {n: np.median(boxes[names == n][:, 3:6], axis=0) for n in np.unique(names)}
    ut["Person"] = np.array([0.70, 0.65, 1.75])  # procedural pedestrians (synth_person BOX_LW / HEIGHT midpoints)
    return real, ut


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--mode", choices=["ratio", "set", "blend"], default="ratio")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--classes", default=None, help="comma-separated subset (default: all with both medians)")
    ap.add_argument("--modes", default=None,
                    help="per-class modes, e.g. 'Car=set,Person=ratio'; only these classes are changed")
    ap.add_argument("--measurement", type=int, default=4)
    ap.add_argument("--print-table", action="store_true")
    args = ap.parse_args()

    real, ut = size_tables(args.measurement)
    classes = set(args.classes.split(",")) if args.classes else set(real) & set(ut)
    per_class = dict(kv.split("=") for kv in args.modes.split(",")) if args.modes else {c: args.mode for c in classes}
    classes = set(per_class)
    if args.print_table or not args.pred:
        for c in sorted(set(real) | set(ut)):
            r, u = real.get(c), ut.get(c)
            ratio = (r / u).round(2) if r is not None and u is not None else "-"
            print(f"{c:<11} real={None if r is None else r.round(2)}  ut={None if u is None else u.round(2)}  ratio={ratio}")
        if not args.pred:
            return

    preds = json.loads(args.pred.read_text())
    for dets in preds.values():
        for d in dets:
            c = d["class"]
            if c not in classes:
                continue
            dims = np.array(d["box"][3:6])
            ratio_dims = dims * real[c] / ut[c]
            new = {"ratio": ratio_dims, "set": real[c],
                   "blend": ratio_dims ** (1 - args.alpha) * real[c] ** args.alpha}[per_class[c]]
            d["box"][2] += (new[2] - dims[2]) / 2
            d["box"][3:6] = [round(float(v), 4) for v in new]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds))
    print(f"wrote {args.out} ({per_class})")


if __name__ == "__main__":
    main()
