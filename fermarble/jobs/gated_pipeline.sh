#!/usr/bin/env bash
# Gated segmentation flow (CPU, detached):
#   1. flow_speed.py --seg: masked flow (PanSR water, fixed geometric bands, droplet filter) with three horizons:
#      m = DPVO, sa = segmentation top edge of water, sb = segmentation water/sky boundary -> runs/flow-gs{,-test}-{m,sa,sb}
#   2. oracle.py --gate-tune: horizon gate (T px / D deg), band minimum N, flow gate (X, Y, Z) fitted on TRAIN
#   3. gated_flow.py: chosen flow per clip (dev, test) -> runs/gated/flow-{dev,test}
#   4. kwide and kwide_crop with the gated flow: dev (official evaluator), test predictions, two bundles
#      (validated, not uploaded), GT-free test speed diagnostics, runs/status_gated.md
# Start: setsid nohup bash fermarble/jobs/gated_pipeline.sh > runs/gated_pipeline.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
G=${G:-runs/gated}; SUFFIX=${SUFFIX:-gseg}; mkdir -p $G   # env overrides: G, GRID, SUFFIX, SKIP_FLOW=1, ONLY_IF_NOT_WORSE=1
log() { echo "$(date '+%F %T') $*"; }
DS="runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
TS="runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe"
VARS="m=mask+fixed+still sa=mask+fixed+still+horizon sb=mask+fixed+still+skyhorizon"
if [ -z "${SKIP_FLOW:-}" ]; then
log "1. masked flow, three horizons"
nice $PY fermarble/flow_speed.py --dataset $TEST --split test --modes wide,flir-VIS --poses $TS --seg runs/seg-pansr \
  --variant $VARS --output runs/flow-gs-test --workers 8 > $G/flow_test.log 2>&1 || log "flow test FAILED"
nice $PY fermarble/flow_speed.py --split train,val --modes wide,flir-VIS --poses $DS --seg runs/seg-pansr \
  --variant $VARS --output runs/flow-gs --workers 8 > $G/flow_dev.log 2>&1 || log "flow dev FAILED"
for v in m sa sb; do log "  $v: dev $(ls runs/flow-gs-$v/speed 2>/dev/null | wc -l)/75, test $(ls runs/flow-gs-test-$v/speed 2>/dev/null | wc -l)/19"; done
fi
log "2. fit the gate on train"
$PY fermarble/oracle.py --split train,val --output $G/tune --gate-tune runs/flow-v2 runs/flow-gs runs/pi3x-dev --workers 8 ${GRID:+--gate-grid $GRID} > $G/tune.log 2>&1 \
  || log "gate tune FAILED"
head -1 $G/tune.log
log "3. gated flow per clip"
$PY fermarble/gated_flow.py --dataset release/release_manifest.json --split train,val --gate $G/tune/gate.json --old runs/flow-v2 \
  --new-prefix runs/flow-gs --output $G/flow-dev 2>&1 | tail -1
$PY fermarble/gated_flow.py --dataset $TEST --split test --gate $G/tune/gate.json --old runs/flow-test \
  --new-prefix runs/flow-gs-test --output $G/flow-test 2>&1 | tail -1
log "4. evaluate and build"
K="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95"
for n in kwide kwide_crop; do
  P=runs/pi3x-dev; PT=runs/pi3x-test; [ $n = kwide_crop ] && P=runs/pi3x-dev-cropmerged && PT=runs/pi3x-test-cropmerged
  ( $PY fermarble/make_predictions.py --split train,val --poses $DS --speed $G/flow-dev --speed-params $G/flow-dev/speed_params.json \
      --vggt $P $K --agree-a runs/vggt-dev --agree-b $P --output $G/dev-$n > $G/dev-$n.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $G/dev-$n >> $G/dev-$n.log 2>&1 \
    || log "dev $n FAILED" ) &
  ( $PY fermarble/make_predictions.py --dataset $TEST --split test --poses $TS --speed $G/flow-test --speed-params $G/flow-test/speed_params.json \
      --vggt $PT $K --agree-a runs/vggt-test --agree-b $PT --output $G/test-$n > $G/test-$n.log 2>&1 || log "test $n FAILED" ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "Gated seg flow vs kwide" --base kwide=runs/yaw-scale/dev-wide:runs/yaw-scale/test-wide \
  --variant kwide_gseg=$G/dev-kwide:$G/test-kwide kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide \
            kwide_crop_gseg=$G/dev-kwide_crop:$G/test-kwide_crop kwide_crop_seg_c=runs/seg-eval/dev-c:runs/seg-eval/test-c \
  --output $G/compare > /dev/null || log "compare FAILED"
$PY fermarble/jobs/compare_variants.py --title "Gated seg flow vs kwide_crop" --base kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide \
  --variant kwide_crop_gseg=$G/dev-kwide_crop:$G/test-kwide_crop --output $G/compare-crop > /dev/null || log "compare crop FAILED"
: > $G/pack.log
for n in kwide kwide_crop; do
  b=runs/submission_test_pi3x_ir_${n}_${SUFFIX}.json
  [ -e $b ] && { echo "$b exists, not overwritten" >> $G/pack.log; continue; }
  if [ -n "${ONLY_IF_NOT_WORSE:-}" ]; then
    base=$G/compare.json; [ $n = kwide_crop ] && base=$G/compare-crop.json
    if ! python3 - $base ${n}_gseg $n <<'PYEOF'
import json, sys
v = json.load(open(sys.argv[1])); new, ref = v[sys.argv[2]]['means'], v[sys.argv[3]]['means']
ok = new['train'] <= ref['train'] + 1e-9 and new['val'] <= ref['val'] + 1e-9
print(f"{sys.argv[2]} vs {sys.argv[3]}: train {new['train']:.3f} vs {ref['train']:.3f}, val {new['val']:.3f} vs {ref['val']:.3f} -> {'build' if ok else 'worse: no bundle'}")
sys.exit(0 if ok else 1)
PYEOF
    then echo "$b not built (worse than ${n} on train or val)" >> $G/pack.log; continue; fi
  fi
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $G/test-$n --output $b >> $G/pack.log 2>&1 || log "pack $n FAILED"
done
$PY fermarble/jobs/seg_speed_diag.py --decisions $G/flow-test/decisions.json --new-dir $G/flow-test \
  --params $G/flow-test/speed_params.json --output $G/test_speed_diag > $G/diag.log 2>&1 || log "diag FAILED"
$PY fermarble/jobs/status_gated.py || log "summary FAILED"
log "DONE"
