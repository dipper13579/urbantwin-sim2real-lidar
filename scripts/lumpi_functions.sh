#!/usr/bin/env bash
# Adopted WINc3 chain; ported from SOURCE_ROOT/setup/win_chain.sh.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
R=$RELEASE_ROOT; L=$WORK_ROOT/outputs/lumpi; D=$WORK_ROOT/data/lumpi_real; V=$WORK_ROOT/outputs/lumpi_realval
FR=$(realpath "${FR:?}"); MR=$(realpath "${MR:?}"); O=${O:?}; mkdir -p "$O"; O=$(realpath "$O")
EXT=(--classes Bus,Truck --alpha 0.5 --grow --l-margin 1.0)
PP=(--relabel "Bus,Truck:10.5"
    --copies "Bus>Truck:0.7,Truck>Bus:0.3,Motorcycle>Bicycle:0.7,Bicycle>Motorcycle:0.5,Person>Bicycle:0.3,Bicycle>Person:0.5"
    --downweight "Truck:5.0:2.3:0.1" --dim-scale "Truck=1,1.14,1.13" --modes "Car=set")
ZS="Person=-0.05;Car=-0.13;Bicycle=-0.10;Motorcycle=-0.10;Bus=-0.10;Truck=-0.10;Van=-0.10;Unknown=-0.10"
ZS_BIG="Car=-0.13;Bus=-0.10;Truck=-0.10;Van=-0.10;Unknown=-0.10"
GROUND=(--classes Person,Bicycle,Motorcycle --offset "Person=-0.03;Bicycle=-0.03;Motorcycle=0.03" --alpha 1.0)
SENS="-9.55,16.3;7.71,-13.68;-19.38,-1.46;19.01,13.54;-13.47,-21.54"
RFP=(--cls Person --alpha 0.5 --iters 2 --radius 0.5 --radial-offset 0.04 --sensors=$SENS)
RFB=(--alpha 0.5 --iters 2 --radius 0.8 --radial-offset 0)
SPEC="Car=fit:0.5;Bus=fit:0.5"; PRULES="--cls Person --nb 4 --wb 0.3 --top 1.0 --wt 0.3"
CARCOPIES="Van>Car:0.7,Bus>Car:0.3,Truck>Car:0.3"; TC="Van>Truck:0.2,Car>Truck:0.1"
D47B="F_v9 F_v11 F_v16 F_v19 F_v22 F_v23 F_v24 F_v25 F_v26 F_v27 F_v30"
say() { echo "$(date '+%H:%M:%S') $*"; }
ochain() {  # raw out_dir -> out_dir/predictions.json, out_dir/car_fit.json (car chain, cap 1000)
  local raw=$1 o=$2
  python tools/refine_extent.py --pred $raw --frames $FR --out $o/ext.json "${EXT[@]}" > /dev/null
  python tools/postprocess.py --pred $o/ext.json --out $o/pp.json "${PP[@]}" --max-per-frame 1000 > /dev/null
  python tools/postprocess.py --pred $o/pp.json --out $o/z.json --z-shift "$ZS" --max-per-frame 1000 > /dev/null
  python tools/refine_points.py --pred $o/z.json --frames $FR --out $o/rp.json "${RFP[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rp.json --frames $FR --out $o/rb.json --cls Bicycle "${RFB[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rb.json --frames $FR --out $o/predictions.json --cls Motorcycle "${RFB[@]}" > /dev/null
  python tools/vert_fit.py --pred $o/predictions.json --frames $FR --out $o/car_fit.json --spec "Car=fit:0.5" > /dev/null
}
pchain() {  # out_dir val_tag raw_suffix cap van_source "person files" [extra --take ...] -> out_dir/final.json, out_dir/final2.json
  local o=$1 tag=$2 rs=$3 cap=$4 van=$5 pf=$6; shift 6; mkdir -p $o
  python tools/ensemble.py --preds $(for m in ${RAW34:-F_v3 F_v4}; do echo -n "$MR/$m/predictions_$rs.json "; done) --out $o/raw34.json --max-per-frame $cap > /dev/null
  python tools/ensemble.py --preds $pf --out $o/person.json --max-per-frame $cap > /dev/null
  python tools/merge_predictions.py --base $o/raw34.json --take Person=$o/person.json "$@" --out $o/raw.json --max-per-frame $cap > /dev/null
  python tools/refine_extent.py --pred $o/raw.json --frames $FR --out $o/ext.json "${EXT[@]}" > /dev/null
  python tools/postprocess.py --pred $o/ext.json --out $o/pp.json "${PP[@]}" --max-per-frame $cap > /dev/null
  python tools/merge_predictions.py --base $o/pp.json --take Van=$van --out $o/pp_van.json --max-per-frame $cap > /dev/null
  mv $o/pp_van.json $o/pp.json
  python tools/postprocess.py --pred $o/pp.json --out $o/zb.json --z-shift "$ZS_BIG" --max-per-frame $cap > /dev/null
  python tools/refine_ground.py --pred $o/zb.json --frames $FR --out $o/z.json "${GROUND[@]}" > /dev/null
  python tools/refine_points.py --pred $o/z.json --frames $FR --out $o/rp.json "${RFP[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rp.json --frames $FR --out $o/rb.json --cls Bicycle "${RFB[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rb.json --frames $FR --out $o/og.json --cls Motorcycle "${RFB[@]}" > /dev/null
  if [ "${STOP_AFTER_OG:-0}" = 1 ]; then return; fi
  python tools/static_prior.py --pred $o/og.json --out $o/s.json --map-preds $V/$tag/M4/og.json $V/$tag/M4r/og.json $V/$tag/M6/og.json $V/$tag/M6r/og.json \
    --classes Person,Bicycle,Motorcycle --q 0.5 --w 0.3 > /dev/null
  python tools/bg_prior.py --pred $o/s.json --frames $FR --out $o/sb.json --map-frames $D/points_M4 $D/points_M4r $D/points_M6 $D/points_M6r \
    --classes Person,Bicycle,Motorcycle --voxel 0.3 --t 0.4 --w 0.3 > /dev/null
  python tools/class_resolve.py --pred $o/sb.json --frames $FR --out $o/sbc.json --lo 0.7 --w 0.5 > /dev/null
  python tools/vert_fit.py --pred $o/sbc.json --frames $FR --out $o/z.json --spec "$SPEC" > /dev/null
  python tools/ped_rules.py --pred $o/z.json --frames $FR --out $o/zr.json $PRULES > /dev/null
  python tools/merge_predictions.py --base $o/zr.json --take Car=$O/A/car_fit.json --out $o/zr8.json --max-per-frame $cap > /dev/null
  python tools/ped_rules.py --pred $o/zr8.json --frames $FR --out $o/w1.json --cls Bicycle --nb 12 --wb 0.3 > /dev/null
  python tools/ped_rules.py --pred $o/w1.json --frames $FR --out $o/final.json --cls Motorcycle --nb 20 --wb 0.3 > /dev/null
  python tools/ped_rules.py --pred $o/final.json --frames $FR --out $o/w3.json --cls Bicycle --top 0.8 --wt 0.3 > /dev/null
  python tools/ped_rules.py --pred $o/w3.json --frames $FR --out $o/final2.json --cls Motorcycle --top 1.0 --wt 0.3 > /dev/null
}
keep() {  # in out classes...: only the boxes of these classes
  python - "$@" <<'EOF'
import json, sys
p = json.load(open(sys.argv[1]))
json.dump({f: [d for d in v if d['class'] in sys.argv[3:]] for f, v in p.items()}, open(sys.argv[2], 'w'))
EOF
}
