#!/usr/bin/env bash
# Extra Omega variants found on dev after omega512_eval.sh: Omega alone on flir-VIS; wide either unchanged (vwwide) or the
# equal mean of Pi3X w24 / w32 / Omega. Waits for the Omega test run, builds test predictions, packs a bundle only for
# variants named on the command line (those that beat the base on train and val), checks 20 finite clips. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/omega512-eval
log() { echo "$(date '+%F %T') $*"; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --speed-params runs/speed-tune3/params_v3a_no_hjump.json"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged --speed runs/flow-test"
R="runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test runs/omega512-test"
declare -A W=( [vis_omega]="0.5,0.5,0,0 wide=0.45,0.45,0.1,0 flir-VIS=0,0,0,1"
               [vis_omega_wmean]="0.5,0.5,0,0 wide=0.3333,0.3333,0,0.3333 flir-VIS=0,0,0,1" )
until grep -q "test done" runs/omega512_run.log; do sleep 30; done
log "test predictions"
for n in vis_omega vis_omega_wmean; do
  ( nice $PY fermarble/make_predictions.py $TS $B --vggt $R --vggt-combine mean --vggt-weights ${W[$n]} --output $O/test-$n > $O/test-$n.log 2>&1 || log "test $n FAILED" ) &
done
wait
for n in "$@"; do
  bn=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_omega512-$n.json
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$n --output $bn > $O/pack-$n.log 2>&1 && log "packed $bn" || { log "pack $n FAILED"; continue; }
  python3 - $bn <<'PYEOF' || log "bundle check FAILED"
import json, sys, numpy as np
b = json.load(open(sys.argv[1]))
base = json.load(open('runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide.json'))
assert len(b) == 20 and all(np.isfinite(np.asarray(v['positions'], float)).all() for v in b.values())
diff = [k for k in base if not np.allclose(np.asarray(base[k]['positions'], float), np.asarray(b[k]['positions'], float))]
print(sys.argv[1], ': 20 clips, all finite; differs from w32mean_vwwide on', len(diff), ':', ', '.join(diff))
PYEOF
done
log "DONE"
