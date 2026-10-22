#!/usr/bin/env bash
# GPU queue (sequential: Pi3X needs ~14 GB, so DPVO must finish first):
#   1. DPVO with CLAHE at focal x0.5 / x0.75, seeds 1 and 2 (seed 0 exists): 5 flir-IR dev clips and
#      test clip_100 -> runs/dpvo-fps8-seed{1,2}-clahe-f{0.5,0.75}, runs/dpvo-test-fps8-seed{1,2}-clahe-f{0.5,0.75}
#   2. Pi3X at phase 0 again, now storing per-window metric displacements (window_disp), on the 80
#      dev clips VGGT covers and the 20 test clips -> runs/pi3x-{dev,test}-metric
#   3. pi3_speed.sh with that speed source -> runs/pi3x-speed-metric (bundle only if it beats flow on train and val)
# Start: setsid nohup bash fermarble/jobs/flirir_metric.sh > runs/flirir_metric.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
PI=$HOME/venvs/pi3/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
V=clip_019,clip_022,clip_024,clip_040,clip_046
log() { echo "$(date '+%F %T') $*"; }
log "1. DPVO flir-IR x0.5/x0.75 seeds 1,2"
for s in 1 2; do for f in 0.5 0.75; do
  d=runs/dpvo-fps8-seed$s-clahe-f$f; mkdir -p $d
  $PY fermarble/run_dpvo.py --clips $V --seed $s --clahe --focal-scale $f --output $d >> $d/run.log 2>&1 &
done; done; wait
for s in 1 2; do for f in 0.5 0.75; do
  d=runs/dpvo-test-fps8-seed$s-clahe-f$f; mkdir -p $d
  $PY fermarble/run_dpvo.py --dataset $TEST --clips clip_100 --seed $s --clahe --focal-scale $f --output $d >> $d/run.log 2>&1 &
done; done; wait
for d in runs/dpvo-fps8-seed{1,2}-clahe-f{0.5,0.75} runs/dpvo-test-fps8-seed{1,2}-clahe-f{0.5,0.75}; do
  echo "  $d: $(ls $d/poses 2>/dev/null | wc -l) clips with poses"; done
log "DPVO DONE"
log "2. Pi3X metric displacements"
mkdir -p runs/pi3x-dev-metric runs/pi3x-test-metric
CLIPS=$(python3 -c "import json;s=json.load(open('runs/vggt-dev/vggt_summary.json'));print(','.join(k for k,v in s.items() if 'error' not in v))")
$PI fermarble/pi3_tool.py --clips $CLIPS --output runs/pi3x-dev-metric >> runs/pi3x-dev-metric/run.log 2>&1 || log "Pi3X dev FAILED"
$PI fermarble/pi3_tool.py --dataset $TEST --split test --output runs/pi3x-test-metric >> runs/pi3x-test-metric/run.log 2>&1 || log "Pi3X test FAILED"
log "PI3X DONE"
log "3. speed evaluation with metric displacements"
FMD=runs/pi3x-dev-metric FMT=runs/pi3x-test-metric O=runs/pi3x-speed-metric BUNDLE=runs/submission_test_pi3x_speed_metric.json \
  bash fermarble/jobs/pi3_speed.sh > runs/pi3_speed_metric.log 2>&1 || log "speed eval FAILED"
log "DONE"
