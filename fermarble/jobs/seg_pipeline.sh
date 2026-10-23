#!/usr/bin/env bash
# PanSR segmentation for flow (detached): GPU segmentation, CPU masked flow in parallel, then re-fit + evaluation.
#   1. seg_run.py: PanSR Swin-L on every 2nd 8 fps flow frame, wide + flir-VIS, test then dev -> runs/seg-pansr
#   2. flow_speed.py --seg, three variants from one flow computation per clip (runs while 1 is going):
#        a = PanSR water mask in the band, b = a + segmentation horizon, c = b + --drop-still
#      -> runs/flow-seg-{a,b,c} (dev), runs/flow-seg-test-{a,b,c} (test)
#   3. flow reliability thresholds re-fitted on TRAIN (oracle --speed-tune, as v3a: best without the
#      horizon-jump gate) for the original flow (control) and a, b, c
#   4. kwide_crop pipeline with each flow + fitted thresholds: dev scored (official evaluator), test
#      predictions; runs/status_seg.md and, only if the best variant is not worse on train and val,
#      runs/submission_test_pi3x_ir_kwide_crop_seg.json (validated, not uploaded)
# Start: setsid nohup bash fermarble/jobs/seg_pipeline.sh > runs/seg_pipeline.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
PS=$HOME/micromamba/envs/pansr/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
E=runs/seg-eval; mkdir -p $E runs/seg-pansr
log() { echo "$(date '+%F %T') $*"; }
DSEEDS="runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
TSEEDS="runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe"
VARS="a=mask b=mask+horizon c=mask+horizon+still"
flow_pass() {
  nice $PY fermarble/flow_speed.py --dataset $TEST --split test --modes wide,flir-VIS --poses $TSEEDS --seg runs/seg-pansr \
    --variant $VARS --output runs/flow-seg-test --workers 6 >> $E/flow_test.log 2>&1
  nice $PY fermarble/flow_speed.py --split train,val --modes wide,flir-VIS --poses $DSEEDS --seg runs/seg-pansr \
    --variant $VARS --output runs/flow-seg --workers 6 >> $E/flow_dev.log 2>&1
}
log "1+2. segmentation (GPU) with masked flow passes (CPU)"
( $PS fermarble/seg_run.py --dataset $TEST --split test --output runs/seg-pansr >> $E/seg_test.log 2>&1
  $PS fermarble/seg_run.py --split train,val --output runs/seg-pansr >> $E/seg_dev.log 2>&1 ) &
SEG=$!
while kill -0 $SEG 2>/dev/null; do flow_pass; sleep 120; done
log "segmentation done: $(ls runs/seg-pansr/*.npz | wc -l) clips"
flow_pass
for v in a b c; do log "  flow $v: dev $(ls runs/flow-seg-$v/speed 2>/dev/null | wc -l)/75, test $(ls runs/flow-seg-test-$v/speed 2>/dev/null | wc -l)/19"; done
log "3. re-fit flow thresholds on train"
TRAIN=$(python3 -c "import json;print(','.join(c['key'] for c in json.load(open('release/release_manifest.json'))['clips'] if c['split']=='train' and c['camera_mode'] in ('wide','flir-VIS')))")
for v in orig a b c; do
  d=runs/flow-seg-$v; [ $v = orig ] && d=runs/flow-v2
  $PY fermarble/oracle.py --clips $TRAIN --speed-tune $d --heading-poses $DSEEDS --output $E/tune-$v > $E/tune-$v.log 2>&1 || log "tune $v FAILED"
  python3 - $E/tune-$v/speed_tune.csv $E/params_$v.json <<'PYEOF'
import csv, json, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if float(r['horizon_jump_max']) >= 1e5]     # as v3a: no horizon-jump gate
best = min(rows, key=lambda r: float(r['mean_med3_flow']))
num = lambda x: float(x) if x not in ('True', 'False', 'cv', 'none') else {'True': True, 'False': False}.get(x, x)
keys = ['agree_tol', 'side_max', 'window_s', 'clamp_lo', 'clamp_hi', 'min_reliable_frac', 'gate_min', 'horizon_jump_max', 'drop_backward', 'gate_stat']
json.dump({k: num(best[k]) for k in keys}, open(sys.argv[2], 'w'), indent=1)
print(sys.argv[2], {k: best[k] for k in keys}, 'train mean med3+flow', best['mean_med3_flow'])
PYEOF
done
log "4. evaluate (kwide_crop pipeline)"
K="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95"
for v in orig a b c; do
  dd=runs/flow-seg-$v; td=runs/flow-seg-test-$v; [ $v = orig ] && dd=runs/flow-v2 && td=runs/flow-test
  ( $PY fermarble/make_predictions.py --split train,val --poses $DSEEDS --speed $dd --speed-params $E/params_$v.json --vggt runs/pi3x-dev-cropmerged \
      $K --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged --output $E/dev-$v > $E/dev-$v.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $E/dev-$v >> $E/dev-$v.log 2>&1 || log "dev $v FAILED" ) &
  ( $PY fermarble/make_predictions.py --dataset $TEST --split test --poses $TSEEDS --speed $td --speed-params $E/params_$v.json --vggt runs/pi3x-test-cropmerged \
      $K --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged --output $E/test-$v > $E/test-$v.log 2>&1 || log "test $v FAILED" ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "Segmentation flow variants vs kwide_crop (re-fitted flow thresholds)" \
  --base kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide \
  --variant refit_only=$E/dev-orig:$E/test-orig a_mask=$E/dev-a:$E/test-a b_mask_horizon=$E/dev-b:$E/test-b \
            c_mask_horizon_still=$E/dev-c:$E/test-c --output $E/compare || log "compare FAILED"
$PY fermarble/jobs/status_seg.py || log "summary FAILED"
log "DONE"
