"""Class-routed merge of predictions.json files: take each class from the run you choose.

  python tools/merge_predictions.py --out outputs/lumpi/C_A_plus_Bperson/predictions.json \
      --base outputs/lumpi/A_cp20/predictions.json --take Person=outputs/lumpi/B_person/predictions.json
"""
from release_paths import KIT_ROOT, WORK_ROOT, RAW_ROOT, PCDET_ROOT
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="predictions used for every class not in --take")
    ap.add_argument("--take", action="append", default=[], help="CLASS=path, repeatable")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-per-frame", type=int, default=200)
    args = ap.parse_args()

    base = json.loads(args.base.read_text())
    routes = {}
    for t in args.take:
        cls, path = t.split("=", 1)
        routes[cls] = json.loads(Path(path).read_text())
    out = {}
    for fid, dets in base.items():
        merged = [d for d in dets if d["class"] not in routes]
        for cls, src in routes.items():
            merged += [d for d in src.get(fid, []) if d["class"] == cls]
        out[fid] = sorted(merged, key=lambda d: -d["score"])[: args.max_per_frame]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    n = sum(len(v) for v in out.values())
    print(f"wrote {args.out}: {len(out)} frames, {n} boxes (routed: {sorted(routes)})")


if __name__ == "__main__":
    main()
