#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export FR=$WORK_ROOT/data/lumpi_real/points_M4 MR=$WORK_ROOT/map_members/M4 O=$WORK_ROOT/map_work
source scripts/lumpi_functions.sh
ogchain() {  # raw frames out_dir [van_source] -> out_dir/og.json
  local raw=$1 fr=$2 o=$3 van=${4:-}; mkdir -p $o
  python tools/refine_extent.py --pred $raw --frames $fr --out $o/ext.json "${EXT[@]}" > /dev/null
  python tools/postprocess.py --pred $o/ext.json --out $o/pp.json "${PP[@]}" --max-per-frame $MAP_CAP > /dev/null
  if [ -n "$van" ]; then
    python tools/merge_predictions.py --base $o/pp.json --take Van=$van --out $o/pp_van.json --max-per-frame $MAP_CAP > /dev/null
    mv $o/pp_van.json $o/pp.json
  fi
  python tools/postprocess.py --pred $o/pp.json --out $o/zb.json --z-shift "$ZS_BIG" --max-per-frame $MAP_CAP > /dev/null
  python tools/refine_ground.py --pred $o/zb.json --frames $fr --out $o/z.json "${GROUND[@]}" > /dev/null
  python tools/refine_points.py --pred $o/z.json --frames $fr --out $o/rp.json "${RFP[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rp.json --frames $fr --out $o/rb.json --cls Bicycle "${RFB[@]}" > /dev/null
  python tools/refine_points.py --pred $o/rb.json --frames $fr --out $o/og.json --cls Motorcycle "${RFB[@]}" > /dev/null
}

for part in 4 4r 6 6r; do
 FR=$WORK_ROOT/data/lumpi_real/points_M$part; MR=$WORK_ROOT/map_members/M$part
 for spec in 'OGP4791116SBCZR8WU u 1000' 'OGProt5SBCZR8Wh2 h 8000' 'OGPtbtbr5SBCZR8Wh2 h 8000'; do
  read -r tag suffix MAP_CAP <<< "$spec"; d=$WORK_ROOT/outputs/lumpi_realval/$tag/M$part; mkdir -p "$d"
  if [ "$suffix" = u ]; then members='F_v4 F_v7 F_v9 F_v11 F_v16'; ps=u; basecap=1000
  else members=$D47B; ps=h2; basecap=4000; fi
  if [ "$tag" = OGProt5SBCZR8Wh2 ]; then
   rot='';for m in $D47B; do rot="$rot ${m}r5 ${m}r-5"; done; members="$members $rot"
  fi
  python tools/ensemble.py --preds "$MR/F_v3/predictions_$suffix.json" "$MR/F_v4/predictions_$suffix.json" --out "$d/base.json" --max-per-frame "$basecap"
  python tools/ensemble.py --preds $(for m in $members; do echo -n "$MR/$m/predictions_$ps.json "; done) --out "$d/person.json" --max-per-frame "$MAP_CAP"
  python tools/merge_predictions.py --base "$d/base.json" --take "Person=$d/person.json" --out "$d/raw.json" --max-per-frame "$MAP_CAP"
  if [ "$tag" = OGPtbtbr5SBCZR8Wh2 ]; then
   mkdir -p "$d/tbm"
   for m in F_v3 F_v4; do
    keep "$MR/$m/predictions_h.json" "$d/tbm/$m.json" Truck Bus
    for a in 5 -5; do keep "$MR/$m/predictions_hr$a.json" "$d/tbm/${m}r$a.json" Truck Bus;done
   done
   python tools/ensemble.py --preds "$d/tbm/F_v3.json" "$d/tbm/F_v4.json" "$d/tbm/F_v3r5.json" "$d/tbm/F_v3r-5.json" "$d/tbm/F_v4r5.json" "$d/tbm/F_v4r-5.json" --out "$d/tb.json" --max-per-frame 4000
   python tools/merge_predictions.py --base "$d/raw.json" --take "Truck=$d/tb.json" --take "Bus=$d/tb.json" --out "$d/raw_tb.json" --max-per-frame "$MAP_CAP"
   cp "$d/raw_tb.json" "$d/raw.json"
  fi
  ogchain "$d/raw.json" "$FR" "$d"
 done
done
