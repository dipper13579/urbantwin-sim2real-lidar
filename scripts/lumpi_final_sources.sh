#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/lumpi_functions.sh"
[ -s $O/predictions.json ] || { echo "base run $O is not finished"; exit 1; }
VO=${VO:?}; mkdir -p $VO; VO=$(realpath $VO)
I=$O/I.json; JP=$O/J/per.json; JF=$O/J/final2.json; KC=$O/K/car_copies.json; LT=$O/L/final2.json

if [ -n "${PERSON:-}" ]; then
  say "J' Person chain, $(echo $PERSON | wc -w) files"
  pchain $VO/J OGProt5SBCZR8Wh2 h 8000 $MR/A_cp20/predictions_h.json "$(for x in $PERSON; do echo -n "$MR/$x/predictions_h2.json "; done)"
  python tools/postprocess.py --pred $VO/J/final.json --out $VO/J/per.json --dim-scale "Person=1.1,1.1,1" --max-per-frame 8000 > /dev/null
  JP=$VO/J/per.json; JF=$VO/J/final2.json
fi
if [ -n "${JFROM:-}" ]; then JP=$JFROM/per.json; JF=$JFROM/final2.json; fi   # stage J of an earlier variant run (prereg 2g)

if [ -n "${CARX:-}" ]; then
  # 11/14 (prereg 2g): CARMEM = the Car members (_u files, rotations 0 / +-5), CARD47 = the members whose _h2 Car boxes are added
  CARMEM=${CARMEM:-F_v3 F_v4 F_v8}; CARD47=${CARD47-$D47B}
  say "K' Car chain ($CARMEM) + Car boxes of ($CARD47)"; k=$VO/K; mkdir -p $k/m
  for M in $CARD47; do keep $MR/$M/predictions_h2.json $k/m/$M.json Car; done
  python tools/ensemble.py --preds $(for m in $CARMEM; do echo -n "$MR/$m/predictions_u.json $MR/$m/predictions_ur5.json $MR/$m/predictions_ur-5.json "; done) \
    $(for M in $CARD47; do echo -n "$k/m/$M.json "; done) --out $k/raw20.json --max-per-frame 1000 > /dev/null
  python tools/merge_predictions.py --base $O/K/raw.json --take Car=$k/raw20.json --out $k/raw.json --max-per-frame 1000 > /dev/null
  ochain $k/raw.json $k
  python tools/postprocess.py --pred $k/car_fit.json --out $k/car_copies.json --copies "${CARCP-$CARCOPIES}" --max-per-frame 1000 > /dev/null
  KC=$k/car_copies.json
fi

if [ -n "${TWB:-}" ]; then
  # 11/14 (prereg 2h): Bicycle / Motorcycle from the U chain (stage B) with other base members than F_v3 F_v4
  say "B' U chain, base members $TWB"
  RAW34="$TWB" pchain $VO/B OGP4791116SBCZR8WU u 1000 $MR/A_cp20/predictions_u.json "$(for x in F_v4 F_v7 F_v9 F_v11 F_v16; do echo -n "$MR/$x/predictions_u.json "; done)"
fi

say "assemble"
python tools/merge_predictions.py --base $I --take Person=$JP --out $VO/pkgJ.json --max-per-frame 8000 > /dev/null
python tools/merge_predictions.py --base $VO/pkgJ.json --take Car=$KC --out $VO/pkgK.json --max-per-frame 8000 > /dev/null
if [ -n "${TWB:-}" ]; then
  python tools/merge_predictions.py --base $VO/pkgK.json --take Bicycle=$VO/B/final2.json --take Motorcycle=$VO/B/final2.json --out $VO/pkgT.json --max-per-frame 8000 > /dev/null
  mv $VO/pkgT.json $VO/pkgK.json
fi
python tools/merge_predictions.py --base $VO/pkgK.json --take Truck=$LT --out $VO/m.json --max-per-frame 8000 > /dev/null
keep $VO/m.json $VO/m.vct.json Van Car Truck
python tools/postprocess.py --pred $VO/m.vct.json --out $VO/m.tc.json --copies "$TC" --max-per-frame 4000 > /dev/null
python tools/merge_predictions.py --base $VO/m.json --take Truck=$VO/m.tc.json --out $VO/predictions.json --max-per-frame 8000 > /dev/null
say "done $VO/predictions.json"
