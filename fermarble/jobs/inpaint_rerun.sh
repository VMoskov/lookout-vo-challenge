#!/usr/bin/env bash
# Static own-boat structure: Pi3X and VGGT on inpainted (+ cropped) keyframes (edits from structure_masks.py).
# Test 103, 106, 112 (Pi3X + VGGT), dev 005, 008, 012, 059 (Pi3X; sanity check of the method).
# Start: setsid nohup bash fermarble/jobs/inpaint_rerun.sh > runs/inpaint_rerun.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PI=$HOME/venvs/pi3/bin/python
VG=$HOME/venvs/vggt/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
ED=runs/structure-masks/edits.json
log() { echo "$(date '+%F %T') $*"; }
mkdir -p runs/pi3x-test-edit runs/vggt-test-edit runs/pi3x-dev-edit
log "Pi3X test"; $PI fermarble/pi3_tool.py --dataset $TEST --clips clip_103,clip_106,clip_112 --edits $ED --output runs/pi3x-test-edit >> runs/pi3x-test-edit/run.log 2>&1 || log "FAILED"
log "VGGT test"; $VG fermarble/vggt_tool.py --dataset $TEST --clips clip_103,clip_106,clip_112 --edits $ED --output runs/vggt-test-edit >> runs/vggt-test-edit/run.log 2>&1 || log "FAILED"
log "Pi3X dev"; $PI fermarble/pi3_tool.py --clips clip_005,clip_008,clip_012,clip_059 --edits $ED --output runs/pi3x-dev-edit >> runs/pi3x-dev-edit/run.log 2>&1 || log "FAILED"
log "DONE"
