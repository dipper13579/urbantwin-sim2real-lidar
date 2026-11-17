"""Inference geometry / calibrated size modules on a finished LUMPI package, in one fixed order, driven by a recipe (JSON).

The final wide_blob, tw_refit, near and k_person settings are selected using labeled, non-forbidden M4w validation frames and box-size statistics, then applied
unchanged to the test package; every frame is processed on its own, from its own points and boxes.

  order     lv_geo -> box_sizes (Truck, Bus) -> wide_blob -> tw_refit -> tw_object (near, k_person) -> box_sizes (Person, Car,
            Bicycle) -> tw_object (share) -> box_sizes (Van) -> van_geo
  recipe    {"lv_geo": "core" | "share", "truck_w": S, "bus_w": S, "wide_blob": true, "tw_refit": true, "near": [T, W],
             "k_person": K, "person_short": [L, W], "car_scale": [SL, SW], "bicycle_h": H, "tw_share": W, "van_w": "V1" | "V2" | "V4",
             "van_ramp": [A, B]}        (missing or null = module off)

  python tools/win_stack.py --pred X.json --frames <dir with frame id .bin> --recipe outputs/lumpi/win/recipe.json --out Y.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path

import box_sizes
import lv_geo
import tw_object
import tw_refit
import van_geo
import wide_blob


def truck_bus(p, frames, r, lv=None):
    """Truck / Bus part.  lv: the lv_geo result if it is already known (cache)."""
    if r.get("lv_geo"):
        p = lv if lv is not None else run_lv(p, frames, r["lv_geo"])
    if r.get("truck_w") or r.get("bus_w"):
        p, _ = box_sizes.apply(p, truck_w=r.get("truck_w"), bus_w=r.get("bus_w"))
    if r.get("wide_blob"):
        p, _ = wide_blob.apply(p, frames, keep=1.0, alpha=0.5)
    return p


LV_PRIOR = dict(r_in=0.793, r_out=0.083)  # share of Bus among long objects inside / outside the bus-like shape cell (validation analysis, "all")


def run_lv(p, frames, mode, recs=None):
    recs = lv_geo.analyse(p, frames) if recs is None else recs
    return lv_geo.strip(lv_geo.build(recs, lv_geo.LONG, lv_geo.MID, rb=True, share=LV_PRIOR if mode == "share" else None))


def two_wheelers(p, frames, r):
    if r.get("tw_refit"):
        p, _, _ = tw_refit.refit(p, frames)
    if r.get("near") or r.get("k_person"):
        p, _ = tw_object.apply(p, near=tuple(r["near"]) if r.get("near") else None, k_person=r.get("k_person"))
    return p


def sizes(p, r):
    if r.get("person_short") or r.get("car_scale") or r.get("bicycle_h"):
        p, _ = box_sizes.apply(p, person_short=tuple(r["person_short"]) if r.get("person_short") else None,
                               car_scale=tuple(r["car_scale"]) if r.get("car_scale") else None, bicycle_h=r.get("bicycle_h"))
    if r.get("tw_share"):
        p, _ = tw_object.apply(p, share=r["tw_share"])
    return p


def vans(p, frames, r):
    if r.get("van_w"):
        p, _ = box_sizes.apply(p, van=r["van_w"])
    if r.get("van_ramp"):
        p = van_geo.rescore(p, van_geo.measure_vans(p, frames), van_geo.FLOOR, tuple(r["van_ramp"]))
    return p


def build(preds, frames, r):
    return vans(sizes(two_wheelers(truck_bus(preds, frames, r), frames, r), r), frames, r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--frames", required=True, help="directory with <frame id>.bin")
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    r = json.loads(Path(args.recipe).read_text())
    preds = json.loads(Path(args.pred).read_text())
    out = build(preds, args.frames, r)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    count = lambda p: sum(len(v) for v in p.values())
    print(f"{args.out}: {len(out)} frames, boxes {count(preds)} -> {count(out)}, recipe {json.dumps({k: v for k, v in r.items() if v})}")


if __name__ == "__main__":
    main()
