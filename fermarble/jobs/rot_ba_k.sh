#!/usr/bin/env bash
# B2/B3: rotation-only BA with Pi3X's own intrinsics (clip median focal / per-window focal) instead of the
# released K; shared SIFT matches; 4 workers (A runs on the GPU meanwhile). Evaluation as rot_ba.sh.
# Start: setsid nohup bash fermarble/jobs/rot_ba_k.sh > runs/rot_ba_k.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
P=runs/rot-ba-k
log() { echo "$(date '+%F %T') $*"; }
for sp in dev test; do log "constraints + refinement, $sp"
  nice -n 15 $PY fermarble/rot_ba.py --split $sp --pi3x runs/pi3x-$sp-automerged --edits runs/structure-masks-auto/edits_$sp.json \
    --output $P/$sp --k-sources pi3x-clip,pi3x-window --workers 4 > $P-$sp.log 2>&1 || log "$sp FAILED"
  grep "refined" $P-$sp.log
done
merge() { rm -rf $3; mkdir -p $3/poses
  for f in $1/poses/*.npz; do b=$(basename $f); if [ -f $2/poses/$b ]; then ln -s $(realpath $2/poses/$b) $3/poses/$b; else ln -s $(realpath $f) $3/poses/$b; fi; done; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-automerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test --agree-b runs/pi3x-test-automerged"
for src in pi3x-clip pi3x-window; do
  O=runs/rot-ba-$src; mkdir -p $O; log "evaluate $src"
  for sp in 0.05 0.1 0.3 1.0; do
    merge runs/pi3x-dev-automerged $P/dev-$src-sp$sp $O/merged-dev-sp$sp
    merge runs/pi3x-test-automerged $P/test-$src-sp$sp $O/merged-test-sp$sp
    ( nice $PY fermarble/make_predictions.py $DS $B --vggt $O/merged-dev-sp$sp --output $O/pred-dev-sp$sp > $O/pred-dev-sp$sp.log 2>&1 \
      && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/pred-dev-sp$sp >> $O/pred-dev-sp$sp.log 2>&1 ) &
    ( nice $PY fermarble/make_predictions.py $TS $B --vggt $O/merged-test-sp$sp --output $O/pred-test-sp$sp > $O/pred-test-sp$sp.log 2>&1 ) &
  done
  wait
  $PY fermarble/jobs/compare_variants.py --title "B ($src intrinsics): rotation-only BA on kwide_crop_inpaint_auto (per sigma_p)" \
    --base base=runs/inpaint-auto-eval/dev:runs/inpaint-auto-eval/test \
    --variant sp0.05=$O/pred-dev-sp0.05:$O/pred-test-sp0.05 sp0.1=$O/pred-dev-sp0.1:$O/pred-test-sp0.1 \
              sp0.3=$O/pred-dev-sp0.3:$O/pred-test-sp0.3 sp1.0=$O/pred-dev-sp1.0:$O/pred-test-sp1.0 --output $O/compare > /dev/null
  python3 - $O $P $src <<'PYEOF'
import json, sys
O, P, src = sys.argv[1:]
v = json.load(open(f'{O}/compare.json'))
best = min((k for k in v if k != 'base'), key=lambda k: v[k]['means']['train'])
dev = json.load(open(f'{P}/dev-{src}-constraints/summary.json')); tst = json.load(open(f'{P}/test-{src}-constraints/summary.json'))
L = [f"sigma_p chosen on TRAIN: **{best}** (train {v[best]['means']['train']:.3f} vs base {v['base']['means']['train']:.3f}; "
     f"val {v[best]['means']['val']:.3f} vs base {v['base']['means']['val']:.3f}) -> "
     + ('beats base on train and val' if v[best]['beats_both'] else 'does NOT beat base on both: no bundle'),
     f"- dev clips refined: {sum(x.get('refined', False) for x in dev.values())}/{len(dev)}; test clips refined: "
     f"{sum(x.get('refined', False) for x in tst.values())}/{len(tst)}",
     f"- test clips whose predictions change ({best}): " + ', '.join(c[5:] for c in (v[best]['test_changed'] or []))]
open(f'{O}/summary.md', 'w').write('\n'.join(L) + '\n')
open(f'{O}/best.txt', 'w').write(best + (' build' if v[best]['beats_both'] else ''))
print('\n'.join(L))
PYEOF
  if grep -q build $O/best.txt; then s=$(cut -d' ' -f1 $O/best.txt); b=runs/submission_test_pi3x_ir_kwide_crop_inpaint_auto_rotba_$src.json
    [ -e $b ] || $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/pred-test-$s --output $b > $O/pack.log 2>&1; fi
  $PY fermarble/jobs/status_last.py > /dev/null
done
log "DONE"
