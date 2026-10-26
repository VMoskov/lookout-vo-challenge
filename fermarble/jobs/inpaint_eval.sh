#!/usr/bin/env bash
# Evaluate the inpainted own-boat structure (after inpaint_rerun.sh): Pi3X dirs = kwide_crop's (cropmerged)
# with the edited clips replaced; GT-free check on test (turns, 60 s RMS vs VGGT original and edited);
# dev drift before/after (official evaluator); test predictions and runs/submission_test_pi3x_ir_kwide_crop_inpaint.json.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/inpaint-eval; mkdir -p $O
for sp in dev test; do m=runs/pi3x-$sp-editmerged/poses; rm -rf runs/pi3x-$sp-editmerged; mkdir -p $m
  for f in runs/pi3x-$sp-cropmerged/poses/*.npz; do b=$(basename $f)
    if [ -f runs/pi3x-$sp-edit/poses/$b ]; then ln -s $(realpath runs/pi3x-$sp-edit/poses/$b) $m/$b; else ln -s $(realpath $f) $m/$b; fi; done
done
for ref in vggt-test vggt-test-edit; do
  $PY fermarble/jobs/flirir_focal_check.py --dataset $TEST --clips clip_103,clip_106,clip_112 --runs orig=runs/pi3x-test-cropmerged \
    edit=runs/pi3x-test-editmerged --vggt runs/$ref --output $O/check_$ref.csv > $O/check_$ref.txt 2>&1
done
K="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --speed-params runs/speed-tune3/params_v3a_no_hjump.json"
$PY fermarble/make_predictions.py --split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 \
  --vggt runs/pi3x-dev-editmerged $K --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-editmerged --output $O/dev > $O/dev.log 2>&1 \
  && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev >> $O/dev.log 2>&1
$PY fermarble/make_predictions.py --dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe \
  --speed runs/flow-test --vggt runs/pi3x-test-editmerged $K --agree-a runs/vggt-test --agree-b runs/pi3x-test-editmerged --output $O/test > $O/test.log 2>&1
echo "test clips changed vs kwide_crop: $(diff -rq $O/test/predictions runs/crop-eval/test-kwide/predictions | grep -o 'clip_[0-9]*.json' | sort -u | tr '\n' ' ')" > $O/changed.txt
B=runs/submission_test_pi3x_ir_kwide_crop_inpaint.json
[ -e $B ] || $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test --output $B > $O/pack.log 2>&1
echo DONE
