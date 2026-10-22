#!/usr/bin/env bash
# Detached evaluation after the flow runs at x1.5/x2.5 and pi3_all.sh finish (it waits for both):
#   A  oracle focal lower bound: pipeline at x1.5/x2.5 (flir-VIS, seed 0), scored, + existing x2.0, x0.5/x0.75
#   B  Pi3X vs VGGT vs DPVO median per camera type (oracle --compare-runs, 80 dev clips VGGT covers)
#   C  switch with Pi3X as source, and VGGT switch with Pi3X tie-breaker (dev, official evaluator; test predictions only)
#   D  clip_075 turn ratios; runs/status_lunch.md (status_summary.py). Nothing is packed or uploaded.
# Start: setsid nohup bash fermarble/jobs/eval_after_pi3.sh > runs/eval_after_pi3.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
DEV=release/release_manifest.json
TEST=lookout-vo-test-v0.1/release_manifest.json
SEEDS="runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
TSEEDS="runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe"
PARAMS="--speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS"
VDEV=clip_000,clip_001,clip_075,clip_076,clip_082,clip_084,clip_088,clip_090
E=runs/pi3x-eval; LB=runs/focal-lb
log() { echo "$(date '+%F %T') $*"; }
mkdir -p $E $LB

log "waiting for flow_speed x1.5/x2.5"
while pgrep -f "[f]low_speed.py --clips $VDEV" > /dev/null; do sleep 60; done
log "A: focal lower bound"
for f in 1.5 2.5; do
  $PY fermarble/make_predictions.py --clips $VDEV --poses runs/dpvo-fps8-seed0-f$f --speed runs/flow-f$f-train $PARAMS \
      --vggt runs/vggt-dev --output $LB/x$f --overwrite > $LB/x$f.log 2>&1 \
    && $PY starter/evaluate.py --dataset $DEV --clips $VDEV --predictions $LB/x$f >> $LB/x$f.log 2>&1 || log "x$f FAILED"
  tail -1 $LB/x$f.log
done
$PY fermarble/oracle.py --split train,val --output $LB/report --focal-lower-bound x1.0=runs/broken-seed/dev \
    x1.5=$LB/x1.5 x2.0=$LB/x2.0 x2.5=$LB/x2.5 x0.5=$LB/x0.5 x0.75=$LB/x0.75 || log "oracle lower bound FAILED"

log "waiting for pi3_all.sh"
while pgrep -f "[p]i3_all.sh" > /dev/null; do sleep 60; done
log "B: heading sources per camera type"
CLIPS=$(python3 -c "import json;s=json.load(open('runs/vggt-dev/vggt_summary.json'));print(','.join(k for k,v in s.items() if 'error' not in v))")
$PY fermarble/oracle.py --clips $CLIPS --output $E/compare --speed-params runs/speed-tune3/params_v3a_no_hjump.json \
    --compare-runs med3=runs/dpvo-fps8-seed0+runs/dpvo-fps8-seed1+runs/dpvo-fps8-seed2:runs/flow-v2 \
    vggt=runs/vggt-dev:runs/flow-v2 pi3x=runs/pi3x-dev:runs/flow-v2 > $E/compare.log 2>&1 || log "compare FAILED"

log "C: Pi3X switch source / tie-breaker"
$PY fermarble/make_predictions.py --split train,val --poses $SEEDS --speed runs/flow-v2 $PARAMS --vggt runs/pi3x-dev \
    --output $E/dev-pi3x-switch --overwrite > $E/dev-pi3x-switch.log 2>&1 \
  && $PY starter/evaluate.py --dataset $DEV --split train,val --predictions $E/dev-pi3x-switch >> $E/dev-pi3x-switch.log 2>&1 || log "dev pi3x switch FAILED"
$PY fermarble/make_predictions.py --split train,val --poses $SEEDS --speed runs/flow-v2 $PARAMS --vggt runs/vggt-dev \
    --tiebreak runs/pi3x-dev --output $E/dev-vggt-tiebreak --overwrite > $E/dev-vggt-tiebreak.log 2>&1 \
  && $PY starter/evaluate.py --dataset $DEV --split train,val --predictions $E/dev-vggt-tiebreak >> $E/dev-vggt-tiebreak.log 2>&1 || log "dev tiebreak FAILED"
$PY fermarble/make_predictions.py --dataset $TEST --split test --poses $TSEEDS --speed runs/flow-test $PARAMS --vggt runs/pi3x-test \
    --output $E/test-pi3x-switch --overwrite > $E/test-pi3x-switch.log 2>&1 || log "test pi3x switch FAILED"
$PY fermarble/make_predictions.py --dataset $TEST --split test --poses $TSEEDS --speed runs/flow-test $PARAMS --vggt runs/vggt-test \
    --tiebreak runs/pi3x-test --output $E/test-vggt-tiebreak --overwrite > $E/test-vggt-tiebreak.log 2>&1 || log "test tiebreak FAILED"
for v in test-pi3x-switch test-vggt-tiebreak; do
  echo "$v: $(diff -rq $E/$v/predictions runs/heading-test-med3-flow-vggtswitch/predictions | grep -o 'clip_[0-9]*.json' | sort -u | sed 's/.json//' | tr '\n' ' ')"
done > $E/test_changes.txt

log "D: clip_075 turn ratios and summary"
$PY - > $E/clip075.txt 2>&1 <<'PYEOF'
import sys; sys.path.insert(0, 'fermarble')
from pathlib import Path
import oracle
from common import load_manifest, select_clips
c = select_clips(load_manifest(Path('release/release_manifest.json')), 'all', 'clip_075')[0]
for lab, dirs in [('DPVO med3 x1', [Path(f'runs/dpvo-fps8-seed{s}') for s in range(3)]),
                  ('DPVO med3 x2', [Path(f'runs/dpvo-fps8-seed{s}-f2') for s in range(3)]),
                  ('VGGT', [Path('runs/vggt-dev')]), ('Pi3X', [Path('runs/pi3x-dev')]),
                  ('Pi3X inject x1', [Path('runs/pi3x-075-k1')]), ('Pi3X inject x2', [Path('runs/pi3x-075-k2')])]:
    print(f"{lab:<16} turn ratio {oracle.heading_scale(Path('release'), c, dirs)['scale']:.2f}")
PYEOF
$PY fermarble/oracle.py --clips clip_075 --output $E/clip075-compare --compare-runs \
    pi3x=runs/pi3x-dev pi3x_k1=runs/pi3x-075-k1 pi3x_k2=runs/pi3x-075-k2 2>/dev/null | grep "^clip_075" >> $E/clip075.txt
$PY fermarble/jobs/status_summary.py || log "summary FAILED"
log "DONE"
