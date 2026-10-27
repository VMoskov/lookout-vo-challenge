#!/usr/bin/env bash
# Speed fixes, step 2 (after speedfix_flow.sh), on both heading bases (w32mean, w32mean_vwwide = vwwide):
#   1. oracle.py --speedfix-tune: fit on TRAIN the horizon sanity rule (U, P, C), the near-band preference
#      (flow none/mask, w_near, reliable or not) and both (mask flow) -> runs/speedfix/tune-<base>
#   2. per family: speed dirs (speedfix_build.py), dev predictions scored (official evaluator), test predictions
#   3. compare_variants vs the base; GT-free speed diagnostics (seg_speed_diag.py) on every test clip whose speed
#      changes (vs runs/speedfix/diag-old.md: the same diagnostics of the current speed); bundles only for variants
#      that beat their base on train and val AND pass the diagnostics
#   4. runs/status_speedfix.md. Never uploads.
# Start: setsid nohup bash fermarble/jobs/speedfix_eval.sh > runs/speedfix_eval.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/speedfix
V3A=runs/speed-tune3/params_v3a_no_hjump.json
log() { echo "$(date '+%F %T') $*"; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --save-heading"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
head_args() { local b=$1 sp=$2 r=$( [ $2 = dev ] && echo runs/vggt-dev || echo runs/vggt-test )
  local p="runs/pi3x-$sp-cropmerged runs/pi3x-$sp-w32"
  [ $b = w32mean ] && echo "--vggt $p --vggt-combine mean" \
    || echo "--vggt $p $r --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"; }
log "1. fit on train"
for b in w32mean vwwide; do
  nice $PY fermarble/oracle.py --split train,val --output $O/tune-$b --speedfix-tune $O/dev-$b-base runs/flow-sf \
    --speed-params $V3A --workers 12 > $O/tune-$b.log 2>&1 || log "tune $b FAILED"
done
log "2. build"
for b in w32mean vwwide; do
  for fam in hs near hs+near; do n=${fam/+/}; S=$O/speed-$b-$n
    $PY fermarble/jobs/speedfix_build.py --best $O/tune-$b/speedfix_best.json --family $fam --output $S > $S.log 2>&1 || log "build $b $fam FAILED"
    ( nice $PY fermarble/make_predictions.py $DS $B $(head_args $b dev) --speed $S-dev --speed-params $S.params.json --output $O/dev-$b-$n > $O/dev-$b-$n.log 2>&1 \
      && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$b-$n >> $O/dev-$b-$n.log 2>&1 \
      || log "dev $b $n FAILED" ) &
    ( nice $PY fermarble/make_predictions.py $TS $B $(head_args $b test) --speed $S-test --speed-params $S.params.json --output $O/test-$b-$n > $O/test-$b-$n.log 2>&1 \
      || log "test $b $n FAILED" ) &
  done
done
wait
log "3. compare, diagnostics, bundles"
for b in w32mean vwwide; do
  $PY fermarble/jobs/compare_variants.py --title "Speed fixes on $b (fitted on train)" --base $b=$O/dev-$b-base:$O/test-$b-base \
    --variant hs=$O/dev-$b-hs:$O/test-$b-hs near=$O/dev-$b-near:$O/test-$b-near hs+near=$O/dev-$b-hsnear:$O/test-$b-hsnear \
    --output $O/compare-$b > /dev/null || log "compare $b FAILED"
  for fam in hs near hs+near; do n=${fam/+/}; S=$O/speed-$b-$n
    CH=$(python3 -c "
import numpy as np
from pathlib import Path
ch=[f.stem for f in sorted(Path('$O/test-$b-$n/heading').glob('*.npz'))
    if not np.allclose(np.load(f)['speed'], np.load(Path('$O/test-$b-base/heading')/f.name)['speed'])]
print(','.join(ch))")
    echo "$CH" > $O/diag-$b-$n.changed
    if [ -n "$CH" ]; then
      $PY fermarble/jobs/seg_speed_diag.py --clips $CH --new-dir $S-test --params $S.params.json --output $O/diag-$b-$n > /dev/null 2>&1 || log "diag $b $n FAILED"
    else echo "No test clip changes speed." > $O/diag-$b-$n.md; fi
    [ -f $O/diag-old.md ] || log "diag-old.md missing (seg_speed_diag on runs/flow-sf-test-none with v3a)"
    if python3 - "$O/compare-$b.json" "$fam" "$O/diag-$b-$n.md" "$O/diag-old.md" <<'PYEOF'
import json, sys
# pass: on every changed test clip with flow speed, no doubling within 1 s, correlation with the old raw flow >= 0.5,
# and no plateau / near-zero stretch the current pipeline's speed does not already have on that clip
cmp_json, fam, new_md, old_md = sys.argv[1:]
def rows(p):
    out = {}
    for l in open(p):
        c = [x.strip() for x in l.split('|')[1:-1]]
        if len(c) == 9 and c[0].isdigit():
            out[c[0]] = dict(old=c[2], new=c[3], flags=c[8])
    return out
kinds = lambda f: {k for k in ('plateau', 'jumps', 'near-zero', 'corr old raw') if k in f}
old, bad = rows(old_md), []
for k, r in rows(new_md).items():
    if r['new'] != 'flow':
        continue
    base = kinds(old[k]['flags']) if k in old and old[k]['new'] == 'flow' else set()
    extra = (kinds(r['flags']) & {'jumps', 'corr old raw'}) | (kinds(r['flags']) - base)
    if extra:
        bad.append(f'{k}: {sorted(extra)}')
c = json.load(open(cmp_json))[fam]
print(fam, 'beats both:', c['beats_both'], 'diagnostic failures:', bad or 'none')
sys.exit(0 if c['beats_both'] and not bad else 1)
PYEOF
    then
      bn=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean$( [ $b = vwwide ] && echo _vwwide )_sf-$n.json
      [ -e $bn ] || { $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$b-$n --output $bn > $O/pack-$b-$n.log 2>&1 \
        && python3 -c "
import json,numpy as np
b=json.load(open('$bn'));assert len(b)==20 and all(np.isfinite(np.asarray(v['positions'],float)).all() for v in b.values());print('20 clips, all finite')" >> $O/pack-$b-$n.log \
        && log "packed $bn" || log "pack $b $n FAILED"; }
    fi
  done
done
python3 fermarble/jobs/status_speedfix.py
log "DONE"
