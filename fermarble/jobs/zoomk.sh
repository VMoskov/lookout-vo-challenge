#!/usr/bin/env bash
# Zoom-aware flir-VIS yaw scale on top of w32mean_vwwide_sf-hsnear: k on flir-VIS clips whose Pi3X focal estimate is
# > T x nominal (make_predictions --zoom-yaw-scale), (k, T) from oracle.py --zoomk-tune (TRAIN). Dev scored, test
# predictions, compare_variants vs the base. Never uploads.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
O=runs/zoomk; mkdir -p $O
SF=runs/speedfix; S=$SF/speed-vwwide-hsnear
ZK=$(python3 -c "import json;b=json.load(open('$O/tune/zoomk_best.json'));print(f\"flir-VIS={b['k']},{b['T']}\")")
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --save-heading"
DS="--split train,val --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 --agree-a runs/vggt-dev --agree-b runs/pi3x-dev-cropmerged"
TS="--dataset $TEST --split test --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --agree-a runs/vggt-test --agree-b runs/pi3x-test-cropmerged"
VWD="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
VWT="--vggt runs/pi3x-test-cropmerged runs/pi3x-test-w32 runs/vggt-test --vggt-combine mean --vggt-weights 0.5,0.5,0 wide=0.45,0.45,0.1"
echo "zoom yaw scale $ZK"
( $PY fermarble/make_predictions.py $DS $B $VWD --speed $S-dev --speed-params $S.params.json --zoom-yaw-scale $ZK --output $O/dev > $O/dev.log 2>&1 \
  && $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev >> $O/dev.log 2>&1 || echo "dev FAILED" ) &
( $PY fermarble/make_predictions.py $TS $B $VWT --speed $S-test --speed-params $S.params.json --zoom-yaw-scale $ZK --output $O/test > $O/test.log 2>&1 || echo "test FAILED" ) &
wait
$PY fermarble/jobs/compare_variants.py --title "Zoom-aware flir-VIS yaw scale ($ZK) on w32mean_vwwide_sf-hsnear" \
  --base hsnear=$SF/dev-vwwide-hsnear:$SF/test-vwwide-hsnear --variant zoomk=$O/dev:$O/test --output $O/compare > /dev/null
echo DONE
