#!/usr/bin/env bash
# CLAHE on thermal (flir-IR): dev 019, 022, 024, 040, 046 and test clip_100. Never touches existing runs or bundles.
#   A. DPVO with CLAHE (same settings as runs/dpvo-*-fps8-seed{0,1,2}-clahe, which the test pose dirs
#      runs/merged-test-seed*-clahe use for clip_100), seeds 3 and 4 -> 5-seed circular median (med5)
#   B. Pi3X and VGGT WITHOUT CLAHE (vggt_tool.py --no-clahe): every existing flir-IR Pi3X / VGGT run already has
#      CLAHE on its keyframes (vggt_tool.py since its first version), so B tests the reverse. Same settings as the
#      runs in use (2 fps, window 24, overlap 4; clip_100 Pi3X with its own-bow crop as pi3x-test-crop), plus one
#      repeat of the CLAHE Pi3X / VGGT run on clip_022 for the run-to-run noise.
#   C. dev variants of the flir-IR heading (make_predictions, vwwide settings, flir-IR clips only), official
#      evaluator + oracle heading-only (saved heading + true speed); test bundle only for the best variant that
#      beats the current handling on dev; runs/status_clahe5.md. Never uploads.
# Start: setsid nohup bash fermarble/jobs/clahe5.sh > runs/clahe5.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
PI=$HOME/venvs/pi3/bin/python
VG=$HOME/venvs/vggt/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
DEV=clip_019,clip_022,clip_024,clip_040,clip_046
O=runs/clahe5; mkdir -p $O
log() { echo "$(date '+%F %T') $*"; }
gpu_free() {   # low priority: wait while another process holds > 2 GB of GPU memory
  while [ "$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '$1 > 2048' | wc -l)" -gt 0 ]; do sleep 60; done; }

log "A. DPVO CLAHE seeds 3, 4"
gpu_free
for s in 3 4; do
  d=runs/dpvo-fps8-seed$s-clahe; mkdir -p $d
  $PY fermarble/run_dpvo.py --clips $DEV --seed $s --clahe --output $d >> $d/run.log 2>&1 || log "DPVO dev seed $s FAILED" &
done
( for s in 3 4; do d=runs/dpvo-test-fps8-seed$s-clahe; mkdir -p $d
    $PY fermarble/run_dpvo.py --dataset $TEST --clips clip_100 --seed $s --clahe --output $d >> $d/run.log 2>&1 || log "DPVO test seed $s FAILED"
  done ) &
wait
log "B. Pi3X / VGGT without CLAHE"
gpu_free
mkdir -p runs/pi3x-dev-noclahe runs/pi3x-test-crop-noclahe runs/vggt-dev-noclahe runs/vggt-test-noclahe runs/pi3x-dev-repeat runs/vggt-dev-repeat
$PI fermarble/pi3_tool.py --clips $DEV --no-clahe --output runs/pi3x-dev-noclahe >> runs/pi3x-dev-noclahe/run.log 2>&1 || log "Pi3X dev FAILED"
$PI fermarble/pi3_tool.py --dataset $TEST --clips clip_100 --crops runs/static-regions/crops.json --no-clahe \
  --output runs/pi3x-test-crop-noclahe >> runs/pi3x-test-crop-noclahe/run.log 2>&1 || log "Pi3X test FAILED"
$PI fermarble/pi3_tool.py --clips clip_022 --output runs/pi3x-dev-repeat >> runs/pi3x-dev-repeat/run.log 2>&1 || log "Pi3X repeat FAILED"
$VG fermarble/vggt_tool.py --clips $DEV --no-clahe --output runs/vggt-dev-noclahe >> runs/vggt-dev-noclahe/run.log 2>&1 || log "VGGT dev FAILED"
$VG fermarble/vggt_tool.py --dataset $TEST --clips clip_100 --no-clahe --output runs/vggt-test-noclahe >> runs/vggt-test-noclahe/run.log 2>&1 || log "VGGT test FAILED"
$VG fermarble/vggt_tool.py --clips clip_022 --output runs/vggt-dev-repeat >> runs/vggt-dev-repeat/run.log 2>&1 || log "VGGT repeat FAILED"

log "C. dev variants"
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --yaw-scale wide=0.95 --save-heading"
HD="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
HT="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
P3="runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
C3="runs/dpvo-fps8-seed0-clahe runs/dpvo-fps8-seed1-clahe runs/dpvo-fps8-seed2-clahe"
C5="$C3 runs/dpvo-fps8-seed3-clahe runs/dpvo-fps8-seed4-clahe"
FM="--agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
FN="--agree-a runs/vggt-dev-noclahe --agree-b runs/pi3x-dev-noclahe"
# label | DPVO pose dirs | agreement rule args (empty = DPVO heading only)
VARIANTS=(
  "current|$P3|--agree-rule 10,15 $FM"
  "dpvo_med3|$P3|"
  "clahe_med3|$C3|"
  "clahe_med5|$C5|"
  "agree_gate_clahe3|$C3|--agree-rule 10,15 $FM"
  "agree_gate_clahe5|$C5|--agree-rule 10,15 $FM"
  "agree_fm_noclahe|$P3|--agree-rule 10,15 $FN"
  "agree_gate_clahe5_fm_noclahe|$C5|--agree-rule 10,15 $FN"
  "fm_always|$P3|--agree-rule 1e9,1e9 $FM"
  "fm_always_noclahe|$P3|--agree-rule 1e9,1e9 $FN"
)
for v in "${VARIANTS[@]}"; do IFS='|' read -r n poses agree <<< "$v"
  ( nice $PY fermarble/make_predictions.py --clips $DEV --poses $poses $B $HD $agree --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --clips $DEV --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 \
    || log "dev $n FAILED" ) &
done
wait
nice $PY fermarble/oracle.py --clips $DEV --output $O/oracle --saved-heading \
  $(for v in "${VARIANTS[@]}"; do n=${v%%|*}; echo -n "$n=$O/dev-$n "; done) > $O/oracle.log 2>&1 || log "oracle FAILED"

log "D. test bundle (best variant that beats current on dev, if any)"
BEST=$($PY fermarble/jobs/status_clahe5.py --pick-best)
log "best variant: ${BEST:-none}"
if [ -n "$BEST" ]; then
  for v in "${VARIANTS[@]}"; do IFS='|' read -r n poses agree <<< "$v"; [ "$n" = "$BEST" ] && break; done
  TP=$(echo "$poses" | sed -e 's#runs/dpvo-fps8-seed\([0-2]\)-clahe#runs/merged-test-seed\1-clahe#g' -e 's#runs/dpvo-fps8-seed\([34]\)-clahe#runs/dpvo-test-fps8-seed\1-clahe#g' \
       -e 's#runs/dpvo-fps8-seed\([0-2]\)\b#runs/dpvo-test-fps8-seed\1#g')
  TA=$(echo "$agree" | sed -e 's#runs/vggt-dev-noclahe#runs/vggt-test-noclahe#' -e 's#runs/pi3x-dev-noclahe#runs/pi3x-test-crop-noclahe#' \
       -e 's#runs/vggt-dev#runs/vggt-test#' -e 's#runs/pi3x-dev-cropmerged#runs/pi3x-test-cropmerged#')
  echo "$TP | $TA" > $O/test-$BEST.args
  nice $PY fermarble/make_predictions.py --dataset $TEST --split test --poses $TP $B $HT $TA --speed runs/flow-test \
    --speed-params runs/speed-tune3/params_v3a_no_hjump.json --output $O/test-$BEST > $O/test-$BEST.log 2>&1 || log "test $BEST FAILED"
  bn=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide_ir-$BEST.json
  [ -e $bn ] || $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$BEST --output $bn > $O/pack-$BEST.log 2>&1 || log "pack FAILED"
  python3 - $bn >> $O/pack-$BEST.log <<'PYEOF' || log "bundle check FAILED"
import json, sys, numpy as np
b = json.load(open(sys.argv[1]))
base = json.load(open('runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide.json'))
assert len(b) == 20 and all(np.isfinite(np.asarray(v['positions'], float)).all() for v in b.values())
diff = [k for k in base if not np.allclose(np.asarray(base[k]['positions'], float), np.asarray(b[k]['positions'], float))]
print('20 clips, all finite; differs from w32mean_vwwide on:', ', '.join(diff) or 'none')
PYEOF
fi
$PY fermarble/jobs/status_clahe5.py
log "DONE"
