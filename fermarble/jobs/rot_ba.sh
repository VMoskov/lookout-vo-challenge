#!/usr/bin/env bash
# B: rotation-only BA on top of kwide_crop_inpaint_auto (CPU). Constraints + refinement for dev and test,
# dev scored per sigma_p, sigma_p chosen on TRAIN, test predicted, bundle only if better on train and val.
# Start: setsid nohup bash fermarble/jobs/rot_ba.sh > runs/rot_ba.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/rot-ba; mkdir -p $O
log() { echo "$(date '+%F %T') $*"; }
log "constraints + refinement, dev"
nice $PY fermarble/rot_ba.py --split dev --pi3x runs/pi3x-dev-automerged --edits runs/structure-masks-auto/edits_dev.json --output $O/dev --workers 10 > $O/dev.log 2>&1 || log "dev FAILED"
tail -1 $O/dev.log
log "constraints + refinement, test"
nice $PY fermarble/rot_ba.py --split test --pi3x runs/pi3x-test-automerged --edits runs/structure-masks-auto/edits_test.json --output $O/test --workers 10 > $O/test.log 2>&1 || log "test FAILED"
tail -1 $O/test.log
merge() {  # base dir, refined dir, merged dir
  rm -rf $3; mkdir -p $3/poses
  for f in $1/poses/*.npz; do b=$(basename $f); if [ -f $2/poses/$b ]; then ln -s $(realpath $2/poses/$b) $3/poses/$b; else ln -s $(realpath $f) $3/poses/$b; fi; done; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-automerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test --agree-b runs/pi3x-test-automerged"
log "dev evaluation per sigma_p"
for sp in 0.05 0.1 0.3 1.0; do
  merge runs/pi3x-dev-automerged $O/dev-sp$sp $O/merged-dev-sp$sp
  merge runs/pi3x-test-automerged $O/test-sp$sp $O/merged-test-sp$sp
  ( $PY fermarble/make_predictions.py $DS $B --vggt $O/merged-dev-sp$sp --output $O/pred-dev-sp$sp > $O/pred-dev-sp$sp.log 2>&1 \
    && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/pred-dev-sp$sp >> $O/pred-dev-sp$sp.log 2>&1 ) &
  ( $PY fermarble/make_predictions.py $TS $B --vggt $O/merged-test-sp$sp --output $O/pred-test-sp$sp > $O/pred-test-sp$sp.log 2>&1 ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "B: rotation-only BA on kwide_crop_inpaint_auto (per sigma_p)" \
  --base base=runs/inpaint-auto-eval/dev:runs/inpaint-auto-eval/test \
  --variant sp0.05=$O/pred-dev-sp0.05:$O/pred-test-sp0.05 sp0.1=$O/pred-dev-sp0.1:$O/pred-test-sp0.1 \
            sp0.3=$O/pred-dev-sp0.3:$O/pred-test-sp0.3 sp1.0=$O/pred-dev-sp1.0:$O/pred-test-sp1.0 --output $O/compare
python3 - <<'PYEOF'
import json
v = json.load(open('runs/rot-ba/compare.json'))
best = min((k for k in v if k != 'base'), key=lambda k: v[k]['means']['train'])
dev = json.load(open('runs/rot-ba/dev-constraints/summary.json')); tst = json.load(open('runs/rot-ba/test-constraints/summary.json'))
L = [f"sigma_p chosen on TRAIN: **{best}** (train {v[best]['means']['train']:.3f} vs base {v['base']['means']['train']:.3f}; "
     f"val {v[best]['means']['val']:.3f} vs base {v['base']['means']['val']:.3f}) -> "
     + ('beats base on train and val' if v[best]['beats_both'] else 'does NOT beat base on both: no bundle'),
     f"- dev clips refined: {sum(x.get('refined', False) for x in dev.values())}/{len(dev)}; "
     f"test clips refined: {sum(x.get('refined', False) for x in tst.values())}/{len(tst)}: "
     + ', '.join(k[5:] for k, x in tst.items() if x.get('refined')),
     f"- test clips whose predictions change ({best}): " + ', '.join(c[5:] for c in (v[best]['test_changed'] or []))]
open('runs/rot-ba/summary.md', 'w').write('\n'.join(L) + '\n')
open('runs/rot-ba/best.txt', 'w').write(best + (' build' if v[best]['beats_both'] else ''))
print('\n'.join(L))
PYEOF
if grep -q build $O/best.txt; then sp=$(cut -d' ' -f1 $O/best.txt); b=runs/submission_test_pi3x_ir_kwide_crop_inpaint_auto_rotba.json
  [ -e $b ] || $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/pred-test-${sp} --output $b > $O/pack.log 2>&1; fi
$PY fermarble/jobs/status_last.py
log "DONE"
