#!/usr/bin/env bash
# Evaluate the overnight Pi3X ensembles (ens_gpu.sh) on top of kwide_crop (11.85 on test; base predictions
# runs/crop-eval/{dev,test}-kwide). Only variants whose Pi3X runs are complete are built (dev scored with the
# official evaluator, test predicted); finished variants with unchanged inputs are skipped, so it can be re-run.
#   1. ensembles of 1 / 1.5 / 2 / 3 / 4 fps: circular mean and median of all 5 and of 1.5 / 2 / 3
#   2. 2 fps window 16 / 32: alone and circular mean with the current run
#   3. weighted circular mean Pi3X / VGGT on wide + flir-VIS (0.9/0.1, 0.8/0.2, 0.7/0.3)
#   4. best mean ensemble of 1-2 (train) with the best weight of 3 (train): members share the Pi3X weight
# The flir-IR agreement rule keeps kwide_crop's Pi3X run. Bundles: the best variant that beats kwide_crop on
# both train and val, and mean5 regardless. Then runs/status_ens.md. Never uploads.
set -u
cd "$(dirname "$0")/../.."
exec 9> runs/ens_eval.lock; flock 9
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/ens-eval; mkdir -p $O
log() { echo "$(date '+%F %T') $*"; }
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
rdir() { [ $2 = vggt ] && echo runs/vggt-$1 || echo runs/pi3x-$1-$2; }
count() { ls $(rdir $1 $2)/poses/*.npz 2>/dev/null | wc -l; }
# all clips done, or the stage finished (a clip Pi3X failed on then uses the remaining runs / DPVO, as in kwide_crop)
complete() { local sp=$1 n=$( [ $1 = dev ] && echo 80 || echo 20 ); shift
  grep -q "STAGE DONE $( [ $sp = dev ] && echo dev-train || echo test )" runs/ens_gpu.log 2>/dev/null && return 0
  for t in "$@"; do [ $(count $sp $t) -ge $n ] || return 1; done; }
throttle() { while [ $(jobs -r | wc -l) -ge 6 ]; do wait -n; done; }
# var NAME "TAGS" [make_predictions args]: builds dev (scored) and test predictions when all TAGS are complete
var() { local n=$1 tags=$2; shift 2; local dv="" tv="" t st
  for t in $tags; do dv+=" $(rdir dev $t)"; tv+=" $(rdir test $t)"; done
  st="$tags $* $(for t in $tags; do echo -n "$(count dev $t)/$(count test $t) "; done)"
  if complete test $tags && ! { [ -f $O/test-$n.stamp ] && [ "$(cat $O/test-$n.stamp)" = "$st" ]; }; then throttle
    ( nice $PY fermarble/make_predictions.py $TS $B --vggt $tv "$@" --output $O/test-$n > $O/test-$n.log 2>&1 \
      && echo "$st" > $O/test-$n.stamp || log "FAILED test $n" ) & fi
  if complete dev $tags && ! { [ -f $O/dev-$n.stamp ] && [ "$(cat $O/dev-$n.stamp)" = "$st" ]; }; then throttle
    ( nice $PY fermarble/make_predictions.py $DS $B --vggt $dv "$@" --output $O/dev-$n > $O/dev-$n.log 2>&1 \
      && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 \
      && echo "$st" > $O/dev-$n.stamp || log "FAILED dev $n" ) & fi; }
ALL5="kf1 kf1.5c cropmerged kf3c kf4"; ALL3="kf1.5c cropmerged kf3c"
log "1. Pi3X keyframe ensembles"
for t in kf1 kf1.5c kf3c kf4; do var $t "$t"; done
var mean5 "$ALL5" --vggt-combine mean; var med5 "$ALL5" --vggt-combine median
var mean3 "$ALL3" --vggt-combine mean; var med3 "$ALL3" --vggt-combine median
log "2. window 16 / 32"
for w in w16 w32; do var $w "$w"; var ${w}mean "cropmerged $w" --vggt-combine mean; done
log "3. Pi3X / VGGT weights"
for w in 0.9 0.8 0.7; do var vw$w "cropmerged vggt" --vggt-combine mean --vggt-weights $w,$(python3 -c "print(round(1-$w,2))"); done
wait
cmp() { local vs="" n
  for n in kf1 kf1.5c kf3c kf4 mean5 med5 mean3 med3 w16 w16mean w32 w32mean vw0.9 vw0.8 vw0.7 best4; do
    [ -f $O/dev-$n.stamp ] && [ -f $O/test-$n.stamp ] && vs+=" $n=$O/dev-$n:$O/test-$n"; done
  [ -n "$vs" ] && $PY fermarble/jobs/compare_variants.py --title "Pi3X ensembles / windows / VGGT weights on kwide_crop (11.85 on test)" \
    --base kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide --variant $vs --output $O/compare > /dev/null; }
cmp
log "4. best mean ensemble + best weight"
# picked on train: best mean ensemble of 1-2 and best Pi3X weight of 3; each ensemble member gets weight/len
P=$(python3 - <<'EOF'
import json
from pathlib import Path
p = Path('runs/ens-eval/compare.json')
c = json.loads(p.read_text()) if p.is_file() else {}
ens = {'mean5': 'kf1 kf1.5c cropmerged kf3c kf4', 'mean3': 'kf1.5c cropmerged kf3c', 'w16mean': 'cropmerged w16',
       'w32mean': 'cropmerged w32'}
e = min((n for n in ens if n in c), key=lambda n: c[n]['means']['train'], default=None)
w = min((n for n in ('vw0.9', 'vw0.8', 'vw0.7') if n in c), key=lambda n: c[n]['means']['train'], default=None)
if e and w and all(n in c for n in ens) and all(n in c for n in ('vw0.9', 'vw0.8', 'vw0.7')):
    tags, wp = ens[e].split(), float(w[2:])
    print(f"{e}+{w}|{ens[e]} vggt|{','.join([f'{wp / len(tags):.6g}'] * len(tags) + [f'{1 - wp:.6g}'])}")
EOF
)
if [ -n "$P" ]; then IFS='|' read -r lbl tags wts <<< "$P"; echo "$lbl" > $O/best4.txt; log "best4 = $lbl (weights $wts)"
  var best4 "$tags" --vggt-combine mean --vggt-weights $wts; wait; cmp
else log "best4 waits for all ensembles and weights"; fi
log "bundles"
pack() { local n=$1 b=runs/submission_test_pi3x_ir_kwide_crop_ens_$1.json
  [ -e $b ] && return; $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$n --output $b >> $O/pack.log 2>&1 \
    && python3 -c "
import json,sys,numpy as np
b=json.load(open('$b'));m=json.load(open('$TEST'))['clips'];bad=[k for k,v in b.items() if not np.isfinite(np.asarray(v['positions'],float)).all()]
assert len(b)==len([c for c in m if c['split']=='test'])==20 and not bad, (len(b),bad)
print('$b: 20 clips, all positions finite')" >> $O/pack.log 2>&1 && log "packed $b" || log "pack FAILED $n"; }
[ -f $O/test-mean5.stamp ] && pack mean5
BEST=$(python3 -c "
import json
from pathlib import Path
p=Path('$O/compare.json');c=json.loads(p.read_text()) if p.is_file() else {}
w=[n for n,v in c.items() if v['beats_both']]
print(min(w,key=lambda n:c[n]['means']['train']) if w else '')")
[ -n "$BEST" ] && pack $BEST
echo "$BEST" > $O/best.txt
python3 fermarble/jobs/status_ens.py
log "eval done"
