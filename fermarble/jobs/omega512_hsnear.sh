#!/usr/bin/env bash
# vis_omega_wmean (Omega alone on flir-VIS, equal Pi3X w24 / w32 / Omega mean on wide) + speed fix sf-hsnear
# (runs/speedfix/speed-vwwide-hsnear, params fitted on train). Zoom-k left out: no gain on dev with Omega on flir-VIS.
# Waits for the Omega test run, builds test predictions, packs and checks the bundle. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/omega512-eval; S=runs/speedfix/speed-vwwide-hsnear; n=vis_wmean_hsnear
log() { echo "$(date '+%F %T') $*"; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
W="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test runs/omega512-test --vggt-combine mean --vggt-weights 0.5,0.5,0,0 wide=0.3333,0.3333,0,0.3333 flir-VIS=0,0,0,1"
until grep -q "test done" runs/omega512_run.log; do sleep 30; done
log "test predictions"
nice $PY fermarble/make_predictions.py $TS $B $W --speed $S-test --speed-params $S.params.json --output $O/test-$n > $O/test-$n.log 2>&1 || { log "test FAILED"; exit 1; }
bn=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_sf-hsnear_omega512-vis_wmean.json
$PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$n --output $bn > $O/pack-$n.log 2>&1 && log "packed $bn" || { log "pack FAILED"; exit 1; }
python3 - $bn <<'PYEOF' || log "bundle check FAILED"
import json, sys, numpy as np
b = json.load(open(sys.argv[1]))
assert len(b) == 20 and all(np.isfinite(np.asarray(v['positions'], float)).all() for v in b.values())
for ref in ['runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_sf-hsnear.json', 'runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_sf-hsnear_zoomk.json']:
    base = json.load(open(ref))
    diff = [k for k in base if not np.allclose(np.asarray(base[k]['positions'], float), np.asarray(b[k]['positions'], float))]
    print('20 clips, all finite; differs from', ref.split('/')[-1], 'on', len(diff), ':', ', '.join(diff))
PYEOF
log "DONE"
