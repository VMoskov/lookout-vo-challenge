#!/usr/bin/env bash
# VGGT weight on wide only, on top of w32mean: wide = weighted circular mean (the two Pi3X runs share 1 - w, VGGT w),
# flir-VIS = w32mean (VGGT weight 0). w = 0.1 / 0.15 / 0.2. Dev scored, test predicted, compared with kwide_crop;
# test clips that differ from w32mean listed. Bundle (best on train among those beating kwide_crop on train and val)
# runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide.json. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/vw-wide; E=runs/ens-eval; mkdir -p $O
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
for w in 0.1 0.15 0.2; do p=$(python3 -c "print(f'{(1-$w)/2:.6g}')"); W="--vggt-combine mean --vggt-weights 0.5,0.5,0 wide=$p,$p,$w"
  ( nice $PY fermarble/make_predictions.py $DS $B --vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev $W --output $O/dev-wide$w > $O/dev-wide$w.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-wide$w >> $O/dev-wide$w.log 2>&1 ) &
  ( nice $PY fermarble/make_predictions.py $TS $B --vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test $W --output $O/test-wide$w > $O/test-wide$w.log 2>&1 ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "VGGT weight on wide only (flir-VIS = w32mean) vs kwide_crop" \
  --base kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide --variant w32mean=$E/dev-w32mean:$E/test-w32mean \
  wide0.1=$O/dev-wide0.1:$O/test-wide0.1 wide0.15=$O/dev-wide0.15:$O/test-wide0.15 wide0.2=$O/dev-wide0.2:$O/test-wide0.2 --output $O/compare
for w in 0.1 0.15 0.2; do echo "wide$w: test clips differing from w32mean: $(diff -rq $O/test-wide$w/predictions $E/test-w32mean/predictions | grep -o 'clip_[0-9]*' | sort -u | tr '\n' ' ')"; done > $O/changed.txt
BEST=$(python3 -c "
import json;c=json.load(open('$O/compare.json'));w=[n for n in c if n.startswith('wide') and c[n]['beats_both']]
print(min(w,key=lambda n:c[n]['means']['train']) if w else '')")
echo "$BEST" > $O/best.txt
if [ -n "$BEST" ]; then b=runs/submission_test_pi3x_ir_kwide_crop_ens_w32mean_vwwide.json
  $PY starter/pack_submission.py --dataset $TEST --split test --predictions $O/test-$BEST --output $b > $O/pack.log 2>&1 && python3 -c "
import json,numpy as np
b=json.load(open('$b'));assert len(b)==20 and all(np.isfinite(np.asarray(v['positions'],float)).all() for v in b.values());print('20 clips, all finite')" >> $O/pack.log; fi
