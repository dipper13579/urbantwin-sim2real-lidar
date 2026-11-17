#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
track=${1:?Choose lumpi or v2x}
case "$track" in
 lumpi)
  edges='-5,-3,-2.5,-2.3,-2.2,-2.1,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.5,2,2.5,3,4,5'
  python tools/realism_match.py --calib "$WORK_ROOT/data/lumpi_real/points_M4w" --exact --xy-cell 1 --z-edges="$edges" --shift=-1.25,0.25,0.04 --noise=0.01,0.01,0.025 --out "$WORK_ROOT/outputs/lumpi/synthetic50_win1m"
  ;;
 v2x)
  edges='-8,-7,-6.5,-6,-5.75,-5.5,-5.25,-5,-4.75,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0,0.5,1,1.5,2'
  python tools/realism_match.py --track v2x --calib "$WORK_ROOT/data/v2x_real/realism_calib" --exact --xy-cell 1 --z-edges="$edges" --ut-frames 9000:9998:20 --out "$WORK_ROOT/outputs/v2x/synthetic50_exact1m"
  ;;
 *) exit 2;;
esac
