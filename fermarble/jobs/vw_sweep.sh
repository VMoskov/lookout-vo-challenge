#!/usr/bin/env bash
# VGGT weight sweep on top of w32mean (circular mean of Pi3X 2 fps window 24 and window 32): weighted circular mean,
# the two Pi3X runs share 1 - w, VGGT gets w (wide + flir-VIS; flir-IR rule unchanged). w = 0 is w32mean and
# w = 0.1 is best4 (runs/ens-eval). Dev scored, test predicted, compared with kwide_crop. No bundles.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/vw-sweep; E=runs/ens-eval; mkdir -p $O
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --speed-params runs/speed-tune3/params_v3a_no_hjump.json --yaw-scale wide=0.95"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --speed runs/flow-v2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --speed runs/flow-test --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
for w in 0.05 0.15 0.2 0.3 0.5; do p=$(python3 -c "print(f'{(1-$w)/2:.6g}')")
  ( nice $PY fermarble/make_predictions.py $DS $B --vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev --vggt-combine mean \
      --vggt-weights $p,$p,$w --output $O/dev-vw$w > $O/dev-vw$w.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-vw$w >> $O/dev-vw$w.log 2>&1 ) &
  ( nice $PY fermarble/make_predictions.py $TS $B --vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test --vggt-combine mean \
      --vggt-weights $p,$p,$w --output $O/test-vw$w > $O/test-vw$w.log 2>&1 ) &
done
wait
$PY fermarble/jobs/compare_variants.py --title "VGGT weight on top of w32mean vs kwide_crop" \
  --base kwide_crop=runs/crop-eval/dev-kwide:runs/crop-eval/test-kwide \
  --variant vw0=$E/dev-w32mean:$E/test-w32mean vw0.05=$O/dev-vw0.05:$O/test-vw0.05 vw0.1=$E/dev-best4:$E/test-best4 \
            vw0.15=$O/dev-vw0.15:$O/test-vw0.15 vw0.2=$O/dev-vw0.2:$O/test-vw0.2 vw0.3=$O/dev-vw0.3:$O/test-vw0.3 \
            vw0.5=$O/dev-vw0.5:$O/test-vw0.5 --output $O/compare
