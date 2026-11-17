#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
test "$(git -C "$PCDET_ROOT" rev-parse HEAD)" = 233f849829b6ac19afb8af8837a0246890908755
cp -R openpcdet_overlay/pcdet/. "$PCDET_ROOT/pcdet/"
echo 'Overlay installed; build OpenPCDet extensions in the activated environment.'
