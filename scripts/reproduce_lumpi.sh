#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export FR=${FR:-$KIT_ROOT/lumpi/detection_test_frames}
export MR=${MR:-$WORK_ROOT/outputs/lumpi}
RUN_ROOT=${RUN_ROOT:-$RELEASE_ROOT/work/replay_lumpi}
mkdir -p "$RUN_ROOT"; RUN_ROOT=$(realpath "$RUN_ROOT")
export O=$RUN_ROOT/base
bash scripts/lumpi_chain.sh
PERSON=$(cat configs/lumpi_person_members.txt) VO=$RUN_ROOT/person bash scripts/lumpi_final_sources.sh
export JFROM=$RUN_ROOT/person/J CARX=1 CARMEM="F_v34 F_v36" CARD47=F_v35
VO=$RUN_ROOT/car bash scripts/lumpi_final_sources.sh
VO=$RUN_ROOT/bicycle TWB="F_v36 F_v34" bash scripts/lumpi_final_sources.sh
for branch in car bicycle; do
  python tools/win_stack.py --pred "$RUN_ROOT/$branch/predictions.json" --frames "$FR" --recipe configs/lumpi_recipe.json --out "$RUN_ROOT/$branch/stack.json"
  python tools/final_rules.py --pred "$RUN_ROOT/$branch/stack.json" --frames "$FR" --truck-van 1.0 --out "$RUN_ROOT/$branch/final.json"
done
python tools/merge_predictions.py --base "$RUN_ROOT/car/final.json" --take "Bicycle=$RUN_ROOT/bicycle/final.json" --out "$RUN_ROOT/predictions.json" --max-per-frame 8000
if [ -n "${DECLARATION:-}" ]; then
  python tools/make_submission.py --track lumpi --synthetic-dir "${SYNTHETIC_DIR:-$WORK_ROOT/outputs/lumpi/synthetic50_win1m}" --predictions "$RUN_ROOT/predictions.json" --declaration "$DECLARATION" --out "$RUN_ROOT/submission_lumpi_WINc3.zip"
fi
