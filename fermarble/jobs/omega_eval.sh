#!/usr/bin/env bash
# VGGT-Omega as a heading source (after omega_run.sh). Dev: oracle heading-only comparison (Omega vs Pi3X w24 / w32,
# VGGT, DPVO 3 seeds) and pipeline variants on top of w32mean_vwwide (speedfix_base.sh settings, official evaluator,
# compare_variants). Test predictions for every variant once runs/omega-test is complete; bundles packed only for
# variants that beat the base on both train and val. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/omega-eval; mkdir -p $O
log() { echo "$(date '+%F %T') $*"; }
until grep -q "^.* test$" runs/omega_run.log; do sleep 60; done
log "1. heading only"
nice $PY fermarble/oracle.py --output $O/oracle --compare-runs pi3x24=runs/pi3x-dev-cropmerged pi3x32=runs/pi3x-dev-w32 vggt=runs/vggt-dev \
  omega=runs/omega-dev dpvo=runs/dpvo-fps8-seed0+runs/dpvo-fps8-seed1+runs/dpvo-fps8-seed2 --workers 12 > $O/oracle.log 2>&1 || log "oracle FAILED"
log "2. pipeline variants"
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --speed-params runs/speed-tune3/params_v3a_no_hjump.json"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged --speed runs/flow-v2"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged --speed runs/flow-test"
# name | heading runs (dev names; -dev -> -test for test) | combine args
VARIANTS=(
  "omega_only|runs/omega-dev|"
  "w32_omega_mean|runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/omega-dev|--vggt-combine mean"
  "vw_omega_wide|runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/omega-dev|--vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
  "w32_omega0.2|runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/omega-dev|--vggt-combine mean --vggt-weights 0.4,0.4,0.2"
)
for v in "${VARIANTS[@]}"; do IFS='|' read -r n runs comb <<< "$v"
  ( nice $PY fermarble/make_predictions.py $DS $B --vggt $runs $comb --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 \
    || log "dev $n FAILED" ) &
done
wait
log "3. test (waits for runs/omega-test)"
until grep -q "dev other modes" runs/omega_run.log; do sleep 60; done
for v in "${VARIANTS[@]}"; do IFS='|' read -r n runs comb <<< "$v"
  ( nice $PY fermarble/make_predictions.py $TS $B --vggt ${runs//-dev/-test} $comb --output $O/test-$n > $O/test-$n.log 2>&1 || log "test $n FAILED" ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "VGGT-Omega heading variants vs w32mean_vwwide" \
  --base vwwide=runs/speedfix/dev-vwwide-base:runs/speedfix/test-vwwide-base \
  --variant $(for v in "${VARIANTS[@]}"; do n=${v%%|*}; echo -n "$n=$O/dev-$n:$O/test-$n "; done) --output $O/compare > /dev/null || log "compare FAILED"
python3 - <<'PYEOF' > $O/beats.txt
import json
c = json.load(open('runs/omega-eval/compare.json'))
print(' '.join(n for n, v in c.items() if n != 'vwwide' and v.get('beats_both')))
PYEOF
for n in $(cat $O/beats.txt); do
  bn=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_omega-$n.json
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$n --output $bn > $O/pack-$n.log 2>&1 && log "packed $bn" || log "pack $n FAILED"
done
log "DONE"
