#!/usr/bin/env bash
# kwide_crop_inpaint_auto variants: (1) wide yaw-scale k re-fit on the cropped/inpainted Pi3X headings,
# (2) + variant-a masked flow (PanSR water, DPVO horizon, v3a thresholds). Dev scored, test predicted.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/last3; mkdir -p $O
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --vggt runs/pi3x-dev-automerged --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-automerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --vggt runs/pi3x-test-automerged --agree-a runs/vggt-test --agree-b runs/pi3x-test-automerged"
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json"
run() { local n=$1; shift
  ( $PY fermarble/make_predictions.py $DS $B "$@" --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 ) &
  ( $PY fermarble/make_predictions.py $TS $B "$@" --output $O/test-$n > $O/test-$n.log 2>&1 ) &
  while [ $(jobs -r | wc -l) -ge 12 ]; do sleep 2; done; }
for k in 0.850 0.875 0.900 0.925 0.950 0.975 1.000 1.025 1.050; do
  run k$k --speed runs/flow-v2 --yaw-scale wide=$k
done
DS=${DS/--split/--speed runs/flow-seg-a --split}; TS=${TS/--split/--speed runs/flow-seg-test-a --split}
run speeda --yaw-scale wide=0.95
wait
echo DONE
