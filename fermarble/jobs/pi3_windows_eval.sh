#!/usr/bin/env bash
# Evaluate the Pi3X keyframe settings on top of kwide_crop_inpaint_auto (dev scored, test predicted, bundles
# only if better on train and val), then refresh runs/status_last.md.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/pi3-windows; mkdir -p $O
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test"
run() { local n=$1 dv=$2 tv=$3 ab=$4 at=$5; shift 5
  ( $PY fermarble/make_predictions.py $DS $B --vggt $dv --agree-b $ab "$@" --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 ) &
  ( $PY fermarble/make_predictions.py $TS $B --vggt $tv --agree-b $at "$@" --output $O/test-$n > $O/test-$n.log 2>&1 ) & }
run kf1.5 runs/pi3x-dev-kf1.5 runs/pi3x-test-kf1.5 runs/pi3x-dev-kf1.5 runs/pi3x-test-kf1.5
run kf3 runs/pi3x-dev-kf3 runs/pi3x-test-kf3 runs/pi3x-dev-kf3 runs/pi3x-test-kf3
run mean3 "runs/pi3x-dev-automerged runs/pi3x-dev-kf1.5 runs/pi3x-dev-kf3" "runs/pi3x-test-automerged runs/pi3x-test-kf1.5 runs/pi3x-test-kf3" \
    runs/pi3x-dev-automerged runs/pi3x-test-automerged --vggt-combine mean
wait
$PY fermarble/jobs/compare_variants.py --title "A: Pi3X keyframe settings on kwide_crop_inpaint_auto" \
  --base base=runs/inpaint-auto-eval/dev:runs/inpaint-auto-eval/test \
  --variant kf1.5=$O/dev-kf1.5:$O/test-kf1.5 kf3=$O/dev-kf3:$O/test-kf3 mean3=$O/dev-mean3:$O/test-mean3 --output $O/compare
for n in kf1.5 kf3 mean3; do
  if python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/compare.json'))['$n']['beats_both'] else 1)"; then
    b=runs/submission_test_pi3x_ir_kwide_crop_inpaint_auto_$n.json
    [ -e $b ] || $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$n --output $b >> $O/pack.log 2>&1
  fi
done
$PY fermarble/jobs/status_last.py
