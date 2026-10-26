#!/usr/bin/env bash
# A: Pi3X at two other keyframe settings on the kwide_crop_inpaint_auto frames (crops.json + automatic
# structure edits): 3 fps (denser, ~8 s windows) and 1.5 fps (sparser, ~16 s windows), window 24 overlap 4.
# Test first, then the 80 dev clips VGGT covers. Then evaluation (pi3_windows_eval.sh) and status.
# Start: setsid nohup bash fermarble/jobs/pi3_windows.sh > runs/pi3_windows.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PI=$HOME/venvs/pi3/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
CR=runs/static-regions/crops.json
log() { echo "$(date '+%F %T') $*"; }
CLIPS=$(python3 -c "import json;s=json.load(open('runs/vggt-dev/vggt_summary.json'));print(','.join(k for k,v in s.items() if 'error' not in v))")
for f in 1.5 3; do d=runs/pi3x-test-kf$f; mkdir -p $d; log "test kf $f"
  $PI fermarble/pi3_tool.py --dataset $TEST --split test --keyframe-fps $f --crops $CR --edits runs/structure-masks-auto/edits_test.json --output $d >> $d/run.log 2>&1 || log "FAILED"; done
for f in 1.5 3; do d=runs/pi3x-dev-kf$f; mkdir -p $d; log "dev kf $f"
  $PI fermarble/pi3_tool.py --clips $CLIPS --keyframe-fps $f --crops $CR --edits runs/structure-masks-auto/edits_dev.json --output $d >> $d/run.log 2>&1 || log "FAILED"; done
log "evaluate"
bash fermarble/jobs/pi3_windows_eval.sh >> runs/pi3_windows.log 2>&1 || log "eval FAILED"
log "DONE"
