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
for tag in OGP4791116SBCZR8WU OGProt5SBCZR8Wh2 OGPtbtbr5SBCZR8Wh2; do for s in 4 4r 6 6r; do [ -s $V/$tag/M$s/og.json ] || { echo "missing $V/$tag/M$s/og.json"; exit 1; }; done; done

# ---- A: car fit (F_v3 F_v4 F_v8 _u)
say "A car fit"; a=$O/A; mkdir -p $a
python tools/ensemble.py --preds $MR/F_v3/predictions_u.json $MR/F_v4/predictions_u.json $MR/F_v8/predictions_u.json --out $a/raw.json --max-per-frame 1000 > /dev/null
ochain $a/raw.json $a

# ---- B + C: U person chain (final2.json = stage C: the two top rules on final.json)
say "B U chain"
pchain $O/B OGP4791116SBCZR8WU u 1000 $MR/A_cp20/predictions_u.json "$(for x in F_v4 F_v7 F_v9 F_v11 F_v16; do echo -n "$MR/$x/predictions_u.json "; done)"
C=$O/B/final2.json

# ---- D: seven-class steps (cap 1000)
say "D class steps"; d=$O/D; mkdir -p $d
python tools/postprocess.py --pred $a/car_fit.json --out $d/car_copies.json --copies "$CARCOPIES" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $C --take Car=$d/car_copies.json --out $d/k.json --max-per-frame 1000 > /dev/null
python tools/postprocess.py --pred $d/k.json --out $d/per.json --dim-scale "Person=1.1,1.1,1" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $d/k.json --take Person=$d/per.json --out $d/d.json --max-per-frame 1000 > /dev/null
python tools/postprocess.py --pred $d/d.json --out $d/bus.json --dim-scale "Bus=1,1.3,1" --z-shift "Bus=0.1" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $d/d.json --take Bus=$d/bus.json --out $d/u.json --max-per-frame 1000 > /dev/null
python tools/postprocess.py --pred $d/u.json --out $d/tru.json --copies "$TC" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $d/u.json --take Truck=$d/tru.json --out $d/t.json --max-per-frame 1000 > /dev/null
python tools/postprocess.py --pred $C --out $d/van.json --copies "Car>Van:0.1" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $d/t.json --take Van=$d/van.json --out $d/v.json --max-per-frame 1000 > /dev/null
python tools/postprocess.py --pred $d/v.json --out $d/x1.cp.json --copies "Truck>Bus:0.3" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $d/v.json --take Bus=$d/x1.cp.json --out $d/x1.json --max-per-frame 1000 > /dev/null

# ---- I: Bus length rule
say "I Bus rule"
python tools/postprocess.py --pred $d/x1.json --out $O/I.json --downweight "Bus:6.5:99:0.3" --max-per-frame 8000 > /dev/null

# ---- J: Person from the d47b members with rotation +-5 (h2)
say "J Person chain"
ROT=""; for M in $D47B; do for r in 5 -5; do ROT="$ROT ${M}r$r"; done; done
pchain $O/J OGProt5SBCZR8Wh2 h 8000 $MR/A_cp20/predictions_h.json "$(for x in $D47B $ROT; do echo -n "$MR/$x/predictions_h2.json "; done)"
python tools/postprocess.py --pred $O/J/final.json --out $O/J/per.json --dim-scale "Person=1.1,1.1,1" --max-per-frame 8000 > /dev/null
python tools/merge_predictions.py --base $O/I.json --take Person=$O/J/per.json --out $O/J/pkg.json --max-per-frame 8000 > /dev/null

# ---- K: Car with rotation +-5
say "K Car chain"; k=$O/K; mkdir -p $k
python tools/ensemble.py --preds $(for m in F_v3 F_v4 F_v8; do echo -n "$MR/$m/predictions_u.json $MR/$m/predictions_ur5.json $MR/$m/predictions_ur-5.json "; done) --out $k/raw.json --max-per-frame 1000 > /dev/null
ochain $k/raw.json $k
python tools/postprocess.py --pred $k/car_fit.json --out $k/car_copies.json --copies "$CARCOPIES" --max-per-frame 1000 > /dev/null
python tools/merge_predictions.py --base $O/J/pkg.json --take Car=$k/car_copies.json --out $k/pkg.json --max-per-frame 8000 > /dev/null

# ---- L: Truck with rotation +-5 (Truck / Bus source of F_v3 F_v4 _h), then the Truck copies step
say "L Truck chain"; mkdir -p $O/TBm
for M in F_v3 F_v4; do
  keep $MR/$M/predictions_h.json $O/TBm/$M.json Truck Bus
  for r in 5 -5; do keep $MR/$M/predictions_hr$r.json $O/TBm/${M}r$r.json Truck Bus; done
done
python tools/ensemble.py --preds $O/TBm/F_v3.json $O/TBm/F_v4.json $O/TBm/F_v3r5.json $O/TBm/F_v3r-5.json $O/TBm/F_v4r5.json $O/TBm/F_v4r-5.json --out $O/TB.json --max-per-frame 4000 > /dev/null
pchain $O/L OGPtbtbr5SBCZR8Wh2 h 8000 $MR/A_cp20/predictions_h.json "$(for x in $D47B; do echo -n "$MR/$x/predictions_h2.json "; done)" --take Truck=$O/TB.json --take Bus=$O/TB.json
python tools/merge_predictions.py --base $k/pkg.json --take Truck=$O/L/final2.json --out $O/m.json --max-per-frame 8000 > /dev/null
keep $O/m.json $O/m.vct.json Van Car Truck
python tools/postprocess.py --pred $O/m.vct.json --out $O/m.tc.json --copies "$TC" --max-per-frame 4000 > /dev/null
python tools/merge_predictions.py --base $O/m.json --take Truck=$O/m.tc.json --out $O/predictions.json --max-per-frame 8000 > /dev/null
say "done $O/predictions.json"
