#!/usr/bin/env bash
# Official VGGT-Omega 1B-512 (facebook/VGGT-Omega, gated; see fermarble/omega_tool.py) on dev wide + flir-VIS, then
# test, with the own-bow crops Pi3X uses. Log markers "test" (dev done) and "test done" are read by omega512_eval.sh.
# Start: setsid nohup bash fermarble/jobs/omega512_run.sh > runs/omega512_run.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
VG=$HOME/venvs/vggt/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
C=runs/static-regions/crops.json
export OMEGA_CKPT=/media/labust/shared3tb_ssd1/macvi5-weights/vggt-omega-official/vggt_omega_1b_512.pt
log() { echo "$(date '+%F %T') $*"; }
mkdir -p runs/omega512-dev runs/omega512-test
DEVC=$(python3 -c "import json;m=json.load(open('release/release_manifest.json'));print(','.join(c['key'] for c in m['clips'] if c['camera_mode'] in ('wide','flir-VIS')))")
log "dev wide + flir-VIS"
$VG fermarble/omega_tool.py --clips $DEVC --crops $C --output runs/omega512-dev >> runs/omega512-dev/run.log 2>&1 || log "dev FAILED"
log "test"
$VG fermarble/omega_tool.py --dataset $TEST --split test --crops $C --output runs/omega512-test >> runs/omega512-test/run.log 2>&1 || log "test FAILED"
log "test done"
