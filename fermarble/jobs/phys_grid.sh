#!/usr/bin/env bash
# Physical priors on kwide_crop_inpaint_auto (dev): global max yaw rate R x max acceleration A grid
# (inf = off); per-camera values are chosen afterwards by composing per-clip results (clips independent).
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
O=runs/phys; mkdir -p $O
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --vggt runs/pi3x-dev-automerged --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-automerged"
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
for R in 2 3 5 7.5 10 15 20 30 45 inf; do for A in 0.05 0.1 0.2 0.3 0.5 inf; do
  n=R${R}_A${A}
  ( $PY fermarble/make_predictions.py $DS $B --max-yaw-rate $R --max-accel $A --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 ) &
  while [ $(jobs -r | wc -l) -ge 14 ]; do sleep 2; done
done; done
wait; echo DONE
