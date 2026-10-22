#!/usr/bin/env bash
# Pi3X speed (CPU) with B's heading (Pi3X heading on wide + flir-VIS, DPVO median elsewhere):
#   const    constant speed
#   flow     flow v3a (= B as built, the base)
#   pi3s<S>  speed from Pi3X camera centres (pose_speed), smoothing sigma S in {2,4,8} s, S fitted on TRAIN
#   combo    flow where flow is reliable (not gated off), Pi3X speed otherwise (sigma as fitted)
# Pi3X speed only on wide + flir-VIS (the flow modes). Dev scored with the official evaluator, test
# predictions built; if a speed variant beats flow on both train and val, the best on train is packed
# as ${BUNDLE:-runs/submission_test_pi3x_speed.json} (validated, not uploaded).
# Start: setsid nohup bash fermarble/jobs/pi3_speed.sh > runs/pi3_speed.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
FMD=${FMD:-runs/pi3x-dev}; FMT=${FMT:-runs/pi3x-test}   # speed source runs (env override)
O=${O:-runs/pi3x-speed}; mkdir -p $O; export O
log() { echo "$(date '+%F %T') $*"; }
B="--speed-params runs/speed-tune3/params_v3a_no_hjump.json --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS"
DEV="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --vggt runs/pi3x-dev"
TST="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --vggt runs/pi3x-test"
run() {  # label, extra args (dev uses runs/pi3x-dev, test runs/pi3x-test for --fm-speed)
  local label=$1; shift
  local dx=${*//@FM@/$FMD} tx=${*//@FM@/$FMT}
  ( $PY fermarble/make_predictions.py $DEV $B $dx --output $O/dev-$label > $O/dev-$label.log 2>&1 \
      && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$label >> $O/dev-$label.log 2>&1 \
      || log "dev $label FAILED" ) &
  ( $PY fermarble/make_predictions.py $TST $B $tx --output $O/test-$label > $O/test-$label.log 2>&1 || log "test $label FAILED" ) &
}
log "constant and Pi3X speed, sigma 2/4/8"
run const --speed-modes none
for s in 2 4 8; do run pi3s$s --speed-modes wide,flir-VIS --fm-speed @FM@ --fm-speed-mode replace --fm-speed-sigma $s; done
wait
S=$(python3 - <<'PYEOF'
import json, os, numpy as np
O = os.environ['O']
m = json.load(open('runs/fm-variants/dev-B/prediction_manifest.json'))['clips']
best = None
for s in (2, 4, 8):
    r = {x['key']: x['drift_pct'] for x in json.load(open(f'{O}/dev-pi3s{s}/results.json'))}
    tr = np.mean([r[k] for k in r if m[k]['split'] == 'train'])
    if best is None or tr < best[1]:
        best = (s, tr)
print(best[0])
PYEOF
)
log "sigma fitted on train: $S s; combo"
run combo --speed-modes wide,flir-VIS --fm-speed @FM@ --fm-speed-mode fallback --fm-speed-sigma $S
wait
$PY fermarble/jobs/compare_variants.py --title "Speed source with B's heading (Pi3X speed sigma fitted on train: $S s)" \
    --base flow=runs/fm-variants/dev-B:runs/fm-variants/test-B-pi3x \
    --variant const=$O/dev-const:$O/test-const pi3s2=$O/dev-pi3s2:$O/test-pi3s2 pi3s4=$O/dev-pi3s4:$O/test-pi3s4 \
              pi3s8=$O/dev-pi3s8:$O/test-pi3s8 combo=$O/dev-combo:$O/test-combo --output $O/compare || log "compare FAILED"
WIN=$(python3 - <<PYEOF
import json
v = json.load(open('$O/compare.json'))
ok = [k for k in ('const', 'pi3s$S', 'combo') if v.get(k, {}).get('beats_both')]   # only train-fitted variants
print(min(ok, key=lambda k: v[k]['means']['train']) if ok else '')
PYEOF
)
if [ -n "$WIN" ]; then
  log "$WIN beats flow on train and val: packing ${BUNDLE:-runs/submission_test_pi3x_speed.json}"
  echo "$WIN" > $O/winner.txt
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$WIN --output ${BUNDLE:-runs/submission_test_pi3x_speed.json} || log "pack FAILED"
else
  log "no speed variant beats flow on both train and val: no bundle"
  echo "none" > $O/winner.txt
fi
$PY fermarble/jobs/status_evening.py || log "summary FAILED"
log "DONE"
