#!/usr/bin/env bash
# Constant speed on all FLIR clips (flir-VIS and flir-IR; flow speed on wide only) on top of w32mean_vwwide and
# w32mean_vwwide_sf-hsnear: same commands as speedfix_base.sh / speedfix_eval.sh with --speed-modes wide. Dev scored
# (official evaluator), test predictions, compare_variants vs each base. Bundles packed by the caller. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/flir-const; mkdir -p $O
SF=runs/speedfix
B="--speed-modes wide --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --save-heading"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
VWD="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
VWT="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
V3A=runs/speed-tune3/params_v3a_no_hjump.json
# NAME DEV_SPEED TEST_SPEED PARAMS
for spec in "vwwide runs/flow-v2 runs/flow-test $V3A" \
            "hsnear $SF/speed-vwwide-hsnear-dev $SF/speed-vwwide-hsnear-test $SF/speed-vwwide-hsnear.params.json"; do
  set -- $spec
  ( nice $PY fermarble/make_predictions.py $DS $B $VWD --speed $2 --speed-params $4 --output $O/dev-$1 > $O/dev-$1.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$1 >> $O/dev-$1.log 2>&1 \
    || echo "dev $1 FAILED" ) &
  ( nice $PY fermarble/make_predictions.py $TS $B $VWT --speed $3 --speed-params $4 --output $O/test-$1 > $O/test-$1.log 2>&1 || echo "test $1 FAILED" ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "Constant speed on FLIR, on w32mean_vwwide" --base vwwide=$SF/dev-vwwide-base:$SF/test-vwwide-base \
  --variant flirconst=$O/dev-vwwide:$O/test-vwwide --output $O/compare-vwwide > /dev/null
$PY fermarble/jobs/compare_variants.py --title "Constant speed on FLIR, on w32mean_vwwide_sf-hsnear" --base hsnear=$SF/dev-vwwide-hsnear:$SF/test-vwwide-hsnear \
  --variant flirconst=$O/dev-hsnear:$O/test-hsnear --output $O/compare-hsnear > /dev/null
echo DONE
