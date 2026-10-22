#!/usr/bin/env bash
# Pi3X keyframe-phase ensemble (GPU), then evaluation (CPU):
#   1. Pi3X on the 80 dev clips VGGT covers and the 20 test clips at keyframe phase 1/3 and 2/3
#      (runs/pi3x-{dev,test}-p1, -p2; phase 0 = the existing runs/pi3x-{dev,test}). Resumable.
#   2. Variant B with the per-frame circular median of the 3 Pi3X runs vs single-run B (official
#      evaluator, dev; test predictions); if it beats single-run B on both train and val, pack
#      runs/submission_test_flow_pi3x3.json (validated, not uploaded).
#   3. runs/status_evening.md (status_evening.py).
# Start: setsid nohup bash fermarble/jobs/pi3_ensemble.sh > runs/pi3_ensemble.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PI=$HOME/venvs/pi3/bin/python
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/pi3x-ensemble
log() { echo "$(date '+%F %T') $*"; }
mkdir -p $O runs/pi3x-dev-p1 runs/pi3x-dev-p2 runs/pi3x-test-p1 runs/pi3x-test-p2
CLIPS=$(python3 -c "import json;s=json.load(open('runs/vggt-dev/vggt_summary.json'));print(','.join(k for k,v in s.items() if 'error' not in v))")
for p in 1 2; do
  ph=$(python3 -c "print($p/3)")
  log "Pi3X dev phase $p/3"
  $PI fermarble/pi3_tool.py --clips $CLIPS --phase $ph --output runs/pi3x-dev-p$p >> runs/pi3x-dev-p$p/run.log 2>&1 || log "dev p$p FAILED"
done
for p in 1 2; do
  ph=$(python3 -c "print($p/3)")
  log "Pi3X test phase $p/3"
  $PI fermarble/pi3_tool.py --dataset $TEST --split test --phase $ph --output runs/pi3x-test-p$p >> runs/pi3x-test-p$p/run.log 2>&1 || log "test p$p FAILED"
done
log "evaluate B with the 3-run Pi3X median"
SP="--speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS --overwrite --vggt-label pi3x3 --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS"
$PY fermarble/make_predictions.py --split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 $SP \
    --vggt runs/pi3x-dev runs/pi3x-dev-p1 runs/pi3x-dev-p2 --output $O/dev-B3 > $O/dev-B3.log 2>&1 \
  && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-B3 >> $O/dev-B3.log 2>&1 || log "dev B3 FAILED"
$PY fermarble/make_predictions.py --dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test $SP \
    --vggt runs/pi3x-test runs/pi3x-test-p1 runs/pi3x-test-p2 --output $O/test-B3 > $O/test-B3.log 2>&1 || log "test B3 FAILED"
$PY fermarble/jobs/compare_variants.py --title "Pi3X 3-phase median vs single-run B" \
    --base B=runs/fm-variants/dev-B:runs/fm-variants/test-B-pi3x --variant B3=$O/dev-B3:$O/test-B3 --output $O/compare || log "compare FAILED"
if python3 -c "import json,sys;sys.exit(0 if json.load(open('$O/compare.json'))['B3']['beats_both'] else 1)"; then
  log "B3 beats B on train and val: packing runs/submission_test_flow_pi3x3.json"
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-B3 --output runs/submission_test_flow_pi3x3.json || log "pack FAILED"
else
  log "B3 does not beat B on both train and val: no bundle"
fi
$PY fermarble/jobs/status_evening.py || log "summary FAILED"
log "DONE"
