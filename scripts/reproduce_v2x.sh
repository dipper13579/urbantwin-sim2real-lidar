#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN_ROOT=${RUN_ROOT:-$RELEASE_ROOT/work/replay_v2x}
python tools/reproduce_v2x.py --out "$RUN_ROOT"
if [ -n "${DECLARATION:-}" ]; then
 python tools/make_submission.py --track v2x --synthetic-dir "${SYNTHETIC_DIR:-$WORK_ROOT/outputs/v2x/synthetic50_exact1m}" --predictions "$RUN_ROOT/predictions.json" --declaration "$DECLARATION" --out "$RUN_ROOT/submission_v2x_G77_H3.zip"
fi
