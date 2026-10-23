#!/usr/bin/env bash
# Rotation-scale calibration of the baseline (pi3x_ir: Pi3X heading on wide + flir-VIS, flir-IR agreement rule):
# dev predictions with the Pi3X heading change multiplied by k, scored with the official evaluator.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
O=runs/yaw-scale; mkdir -p $O
B="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --vggt runs/pi3x-dev --speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev"
for k in ${KS:-0.70 0.75 0.80 0.85 0.90 0.95 1.00 1.05 1.10 1.15 1.20 1.25 1.30}; do
  ( $PY fermarble/make_predictions.py $B --yaw-scale $k --output $O/k$k > $O/k$k.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/k$k >> $O/k$k.log 2>&1 ) &
  while [ $(jobs -r | wc -l) -ge 7 ]; do sleep 2; done
done
wait
