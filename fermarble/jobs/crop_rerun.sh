#!/usr/bin/env bash
# Own-bow / deck crop (crops from fermarble/jobs/static_regions.py -> runs/static-regions/crops.json):
#   1. DPVO with the bottom crop on the frozen-DPVO test clips 096, 104 (flir-VIS) and 100 (flir-IR, + CLAHE), 3 seeds
#   2. Pi3X with the bottom crop on the cropped dev and test clips -> runs/pi3x-{dev,test}-crop
#   3. merged Pi3X dirs (cropped where available, else the original run) and the baseline (pi3x_ir) rebuilt
#      with them: dev scored (official evaluator) and compared; test predictions; bundle only if it beats
#      the baseline on both train and val. GT-free DPVO tracking check for the cropped test clips.
# Start: setsid nohup bash fermarble/jobs/crop_rerun.sh > runs/crop_rerun.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
PI=$HOME/venvs/pi3/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
C=runs/static-regions/crops.json
O=runs/crop-eval; mkdir -p $O
log() { echo "$(date '+%F %T') $*"; }
crop() { python3 -c "import json;print(json.load(open('$C'))['$1'])"; }
log "1. DPVO cropped on 096, 104, 100"
for s in 0 1 2; do
  for k in clip_096 clip_104; do d=runs/dpvo-test-fps8-seed$s-crop-$k; mkdir -p $d
    $PY fermarble/run_dpvo.py --dataset $TEST --clips $k --seed $s --crop-bottom $(crop $k) --output $d >> $d/run.log 2>&1 & done
  d=runs/dpvo-test-fps8-seed$s-clahe-crop-clip_100; mkdir -p $d
  $PY fermarble/run_dpvo.py --dataset $TEST --clips clip_100 --seed $s --clahe --crop-bottom $(crop clip_100) --output $d >> $d/run.log 2>&1 &
  wait
done
log "2. Pi3X cropped"
DEVC=$(python3 -c "import json;m=json.load(open('release/release_manifest.json'));c=json.load(open('$C'));print(','.join(x['key'] for x in m['clips'] if x['key'] in c))")
TSTC=$(python3 -c "import json;m=json.load(open('$TEST'));c=json.load(open('$C'));print(','.join(x['key'] for x in m['clips'] if x['key'] in c))")
mkdir -p runs/pi3x-dev-crop runs/pi3x-test-crop
$PI fermarble/pi3_tool.py --clips $DEVC --crops $C --output runs/pi3x-dev-crop >> runs/pi3x-dev-crop/run.log 2>&1 || log "Pi3X dev FAILED"
$PI fermarble/pi3_tool.py --dataset $TEST --clips $TSTC --crops $C --output runs/pi3x-test-crop >> runs/pi3x-test-crop/run.log 2>&1 || log "Pi3X test FAILED"
log "3. evaluate"
for sp in dev test; do m=runs/pi3x-$sp-cropmerged/poses; mkdir -p $m
  for f in runs/pi3x-$sp/poses/*.npz; do b=$(basename $f)
    if [ -f runs/pi3x-$sp-crop/poses/$b ]; then ln -sf ../../pi3x-$sp-crop/poses/$b $m/$b; else ln -sf ../../pi3x-$sp/poses/$b $m/$b; fi; done
done
B="--speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15"
$PY fermarble/make_predictions.py --split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 \
    --vggt runs/pi3x-dev-cropmerged $B --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged --output $O/dev > $O/dev.log 2>&1 \
  && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev >> $O/dev.log 2>&1 || log "dev FAILED"
$PY fermarble/make_predictions.py --dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test \
    --vggt runs/pi3x-test-cropmerged $B --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged --output $O/test > $O/test.log 2>&1 || log "test FAILED"
$PY fermarble/jobs/compare_variants.py --title "Own-bow crop for Pi3X vs baseline (pi3x_ir, 12.56 on test)" \
    --base baseline=runs/ir-agree/dev:runs/ir-agree/test --variant crop=$O/dev:$O/test --output $O/compare || log "compare FAILED"
if python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/compare.json'))['crop']['beats_both'] else 1)"; then
  log "crop beats the baseline on train and val: packing runs/submission_test_pi3x_ir_crop.json"
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test --output runs/submission_test_pi3x_ir_crop.json || log "pack FAILED"
else log "crop does not beat the baseline on both train and val: no bundle"; fi
for k in clip_096 clip_104; do
  $PY fermarble/jobs/flirir_focal_check.py --dataset $TEST --clips $k --vggt runs/vggt-test --pi3x runs/pi3x-test \
    --runs orig=runs/merged-test-seed0-clahe+runs/merged-test-seed1-clahe+runs/merged-test-seed2-clahe \
           crop=runs/dpvo-test-fps8-seed0-crop-$k+runs/dpvo-test-fps8-seed1-crop-$k+runs/dpvo-test-fps8-seed2-crop-$k --output $O/dpvo_$k.csv
done > $O/dpvo_check.txt 2>&1
$PY fermarble/jobs/flirir_focal_check.py --dataset $TEST --clips clip_100 --vggt runs/vggt-test --pi3x runs/pi3x-test \
    --runs orig=runs/merged-test-seed0-clahe+runs/merged-test-seed1-clahe+runs/merged-test-seed2-clahe \
           crop=runs/dpvo-test-fps8-seed0-clahe-crop-clip_100+runs/dpvo-test-fps8-seed1-clahe-crop-clip_100+runs/dpvo-test-fps8-seed2-clahe-crop-clip_100 \
    --output $O/dpvo_clip_100.csv >> $O/dpvo_check.txt 2>&1
log "DONE"
