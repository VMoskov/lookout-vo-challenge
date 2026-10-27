#!/usr/bin/env bash
# Speed fixes: the two heading bases (w32mean, w32mean_vwwide) rebuilt with --save-heading (final heading + speed per
# clip, for the oracle's speed fit), dev and test. Predictions must equal runs/ens-eval/*-w32mean and runs/vw-wide/*-wide0.1.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/speedfix; mkdir -p $O
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --save-heading"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
SP="--speed-params ${SPEED_PARAMS:-runs/speed-tune3/params_v3a_no_hjump.json}"
W32D="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 --vggt-combine mean"
W32T="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 --vggt-combine mean"
VWD="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
VWT="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
# NAME DEV_SPEED TEST_SPEED [extra]: both bases
build() { local n=$1 ds=$2 ts=$3; shift 3
  for b in w32mean vwwide; do [ $b = w32mean ] && { hd=$W32D; ht=$W32T; } || { hd=$VWD; ht=$VWT; }
    ( nice $PY fermarble/make_predictions.py $DS $B $SP $hd --speed $ds "$@" --output $O/dev-$b-$n > $O/dev-$b-$n.log 2>&1 \
      && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$b-$n >> $O/dev-$b-$n.log 2>&1 ) &
    ( nice $PY fermarble/make_predictions.py $TS $B $SP $ht --speed $ts "$@" --output $O/test-$b-$n > $O/test-$b-$n.log 2>&1 ) &
  done; wait; }
build ${NAME:-base} ${DEV_SPEED:-runs/flow-v2} ${TEST_SPEED:-runs/flow-test}
