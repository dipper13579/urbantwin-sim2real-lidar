#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# Place the licensed Measurement4/6 Label.csv in WORK_ROOT/data/lumpi_real/M{4,6}_Label.csv first.
python tools/select_lumpi_frames.py base
for m in 4 6; do
 python tools/fetch_ut.py fetch --url "https://data.uni-hannover.de/vault/ikg/busch/LUMPI/Measurement$m/lidar.zip" --names-file "$WORK_ROOT/selection/lumpi_M$m.txt" --out "$RAW_ROOT/LUMPI-real/M$m" --workers 6
done
python tools/prepare_lumpi_real.py
python tools/select_lumpi_frames.py rare
for m in 4 6; do
 python tools/fetch_ut.py fetch --url "https://data.uni-hannover.de/vault/ikg/busch/LUMPI/Measurement$m/lidar.zip" --names-file "$WORK_ROOT/selection/rare_names_m$m.txt" --out "$RAW_ROOT/LUMPI-real/M${m}r" --workers 6
done
python tools/prepare_lumpi_rare.py
python tools/select_lumpi_frames.py window
python tools/fetch_ut.py fetch --url https://data.uni-hannover.de/vault/ikg/busch/LUMPI/Measurement4/lidar.zip --names-file "$WORK_ROOT/selection/lumpi_M4w.txt" --out "$RAW_ROOT/LUMPI-real/M4w" --workers 6
python tools/prepare_lumpi_win.py
