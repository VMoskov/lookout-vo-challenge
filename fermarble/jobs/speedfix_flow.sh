#!/usr/bin/env bash
# Speed fixes, step 1 (CPU): one flow computation per wide / flir-VIS clip with the variants
#   none (= flow-v2 / flow-test), mask (PanSR water only), hsU (horizon sanity candidates: segmentation horizon
#   shifted up by U px, U = 0..20) and mask+hsU. Test first, then dev. Outputs runs/flow-sf{,-test}-<variant>.
# Start: setsid nohup bash fermarble/jobs/speedfix_flow.sh > runs/speedfix_flow.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
log() { echo "$(date '+%F %T') $*"; }
V="none=none mask=mask"
for u in 0 5 10 15 20; do V+=" hs$u=hs$u mhs$u=mask+hs$u"; done
log "test"
nice $PY fermarble/flow_speed.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --modes wide,flir-VIS \
  --poses runs/dpvo-test-fps8-seed0 --seg runs/seg-pansr --variant $V --output runs/flow-sf-test --workers 8 > runs/flow-sf-test.log 2>&1 || log "test FAILED"
log "dev"
nice $PY fermarble/flow_speed.py --split train,val --modes wide,flir-VIS --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 \
  --seg runs/seg-pansr --variant $V --output runs/flow-sf --workers 8 > runs/flow-sf.log 2>&1 || log "dev FAILED"
log "DONE"
