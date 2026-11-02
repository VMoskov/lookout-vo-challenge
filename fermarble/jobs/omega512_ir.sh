#!/usr/bin/env bash
# Official VGGT-Omega 512 on the dev flir-IR clips (CLAHE keyframes, as vggt_tool / pi3_tool), then flir-IR variants of the
# agreement rule with Omega in place of old VGGT or Pi3X (vis_wmean_hsnear settings, official evaluator, flir-IR clips).
# Test clip_100 already has Omega poses in runs/omega512-test. Never uploads.
set -u
cd "$(dirname "$0")/../.."
VG=$HOME/venvs/vggt/bin/python; PY=$HOME/micromamba/envs/dpvo/bin/python
export OMEGA_CKPT=/media/labust/shared3tb_ssd1/macvi5-weights/vggt-omega-official/vggt_omega_1b_512.pt
O=runs/omega512-eval; OD=runs/omega512-dev-ir; mkdir -p $OD
DEV=clip_019,clip_022,clip_024,clip_040,clip_046
log() { echo "$(date '+%F %T') $*"; }
log "Omega dev flir-IR"
$VG fermarble/omega_tool.py --clips $DEV --crops runs/static-regions/crops.json --output $OD >> $OD/run.log 2>&1 || { log "Omega FAILED"; exit 1; }
log "variants"
S=runs/speedfix/speed-vwwide-hsnear
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 10,15 --yaw-scale wide=0.95 --speed $S-dev --speed-params $S.params.json"
DS="--clips $DEV --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2"
W="--vggt runs/pi3x-dev-cropmerged runs/pi3x-dev-w32 runs/vggt-dev runs/omega512-dev --vggt-combine mean --vggt-weights 0.5,0.5,0,0 wide=0.3333,0.3333,0,0.3333 flir-VIS=0,0,0,1"
for v in "ir_current|runs/vggt-dev|runs/pi3x-dev-cropmerged" "ir_omega_pi3x|$OD|runs/pi3x-dev-cropmerged" "ir_vggt_omega|runs/vggt-dev|$OD"; do
  IFS='|' read -r n a b <<< "$v"
  ( nice $PY fermarble/make_predictions.py $DS $B $W --agree-a $a --agree-b $b --output $O/dev-$n > $O/dev-$n.log 2>&1 \
    && nice $PY starter/evaluate.py --dataset release/release_manifest.json --clips $DEV --predictions $O/dev-$n >> $O/dev-$n.log 2>&1 || log "$n FAILED" ) &
done
wait
log "DONE"
