#!/usr/bin/env bash
# VGGT-Omega (mapvggt abl_base_best, see fermarble/omega_tool.py) on all clips with the own-bow crops Pi3X uses:
# dev wide + flir-VIS first (the modes with a Pi3X heading), then test, then the remaining dev clips.
# Start: setsid nohup bash fermarble/jobs/omega_run.sh > runs/omega_run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
VG=$HOME/venvs/vggt/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
C=runs/static-regions/crops.json
log() { echo "$(date '+%F %T') $*"; }
clips() { python3 -c "import json;m=json.load(open('$1'));print(','.join(c['key'] for c in m['clips'] if c['split'] in '$2'.split(',') and (c['camera_mode'] in '$3'.split(',')) == $4))"; }
mkdir -p runs/omega-dev runs/omega-test
log "dev wide + flir-VIS"
$VG fermarble/omega_tool.py --clips $(clips release/release_manifest.json train,val wide,flir-VIS True) --crops $C --output runs/omega-dev >> runs/omega-dev/run.log 2>&1 || log "dev 1 FAILED"
log "test"
$VG fermarble/omega_tool.py --dataset $TEST --split test --crops $C --output runs/omega-test >> runs/omega-test/run.log 2>&1 || log "test FAILED"
log "dev other modes"
$VG fermarble/omega_tool.py --clips $(clips release/release_manifest.json train,val wide,flir-VIS False) --crops $C --output runs/omega-dev >> runs/omega-dev/run.log 2>&1 || log "dev 2 FAILED"
log "DONE"
