#!/usr/bin/env bash
# flir-IR heading from Omega alone (agreement rule forced on with Omega as both members), after omega512_ir.sh.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
O=runs/omega512-eval; OD=runs/omega512-dev-ir; n=ir_omega_only
DEV=clip_019,clip_022,clip_024,clip_040,clip_046
until grep -q "variants\|FAILED" runs/omega512_ir.log; do sleep 10; done
S=runs/speedfix/speed-vwwide-hsnear
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 1e9,1e9 --yaw-scale wide=0.95 --speed $S-dev --speed-params $S.params.json"
DS="--clips $DEV --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
W="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev runs/omega512-dev --vggt-combine mean --vggt-weights 0.5,0.5,0,0 wide=0.3333,0.3333,0,0.3333 flir-VIS=0,0,0,1"
nice $PY fermarble/make_predictions.py $DS $B $W --agree-a $OD --agree-b $OD --output $O/dev-$n > $O/dev-$n.log 2>&1 \
  && nice $PY starter/evaluate.py --dataset release/release_manifest.json --clips $DEV --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 && echo "$n DONE" || echo "$n FAILED"
