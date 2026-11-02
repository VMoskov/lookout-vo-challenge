#!/usr/bin/env bash
# Full rerun of the final entry's test pipeline from the videos, into new dirs under $R (default runs/rerun-test),
# with the crops of runs/static-regions/crops_v2.json, then final_entry.sh on those runs. Settings as in the runs
# behind the entry:
#   DPVO 8 fps seeds 0/1/2, plus a CLAHE run of the thermal clips per seed
#   Pi3X w24 uncropped and cropped, Pi3X w32 (cropped), VGGT; keyframe CLAHE on thermal clips (the default)
#   Omega 512 (cropped, keyframe CLAHE on thermal clips, as when it ran)
#   PanSR segmentation; flow on the seed-0 DPVO run, variants mask and mask+hs0 (the ones the fitted speed fix uses)
#   speed fix hs+near from the fitted candidate runs/speedfix/tune-vwwide/speedfix_best.json (not re-fitted)
# GPU stages run one at a time; flow (CPU) runs once DPVO seed 0 and the segmentation are done. The tools skip
# finished clips, so a restart resumes. Never uploads.
# Start: setsid nohup bash fermarble/jobs/rerun_test.sh > runs/rerun-test.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
PI=$HOME/venvs/pi3/bin/python
VG=$HOME/venvs/vggt/bin/python
PS=$HOME/micromamba/envs/pansr/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
C=runs/static-regions/crops_v2.json
R=${R:-runs/rerun-test}
export OMEGA_CKPT=/media/labust/shared3tb_ssd1/macvi5-weights/vggt-omega-official/vggt_omega_1b_512.pt
log() { echo "$(date '+%F %T') $*"; }
gpu() { local name=$1 dir=$2; shift 2; mkdir -p $dir; log "start $name"
  "$@" --output $dir >> $dir/run.log 2>&1 && log "done $name" || log "FAILED $name"; }
mkdir -p $R
THERMAL=$($PY -c "import json,sys;sys.path.insert(0,'fermarble');import thermal
print(','.join(c['key'] for c in json.load(open('$TEST'))['clips'] if c['camera_mode'] in thermal.THERMAL_MODES))")
CROPPED=$(python3 -c "import json;c=json.load(open('$C'))
print(','.join(x['key'] for x in json.load(open('$TEST'))['clips'] if x['key'] in c))")
log "thermal clips: $THERMAL; cropped clips: $CROPPED"
gpu "DPVO seed 0" $R/dpvo-seed0 $PY fermarble/run_dpvo.py --dataset $TEST --split test --seed 0
gpu "PanSR segmentation" $R/seg-pansr $PS fermarble/seg_run.py --dataset $TEST --split test
( log "start flow"
  nice $PY fermarble/flow_speed.py --dataset $TEST --split test --modes wide,flir-VIS --poses $R/dpvo-seed0 --seg $R/seg-pansr \
    --variant mask=mask mhs0=mask+hs0 --output $R/flow-sf-test --workers 8 > $R/flow.log 2>&1 && log "done flow" || log "FAILED flow" ) &
FLOW=$!
for s in 1 2; do gpu "DPVO seed $s" $R/dpvo-seed$s $PY fermarble/run_dpvo.py --dataset $TEST --split test --seed $s; done
for s in 0 1 2; do gpu "DPVO seed $s CLAHE" $R/dpvo-seed$s-clahe $PY fermarble/run_dpvo.py --dataset $TEST --clips $THERMAL --seed $s --clahe; done
gpu "Pi3X w24" $R/pi3x $PI fermarble/pi3_tool.py --dataset $TEST --split test
gpu "Pi3X w24 cropped" $R/pi3x-crop $PI fermarble/pi3_tool.py --dataset $TEST --clips $CROPPED --crops $C
gpu "Pi3X w32" $R/pi3x-w32 $PI fermarble/pi3_tool.py --dataset $TEST --split test --window 32 --crops $C
gpu "VGGT" $R/vggt $VG fermarble/vggt_tool.py --dataset $TEST --split test
gpu "Omega" $R/omega512 $VG fermarble/omega_tool.py --dataset $TEST --split test --crops $C
wait $FLOW
log "speed fix"
$PY fermarble/jobs/speedfix_build.py --best runs/speedfix/tune-vwwide/speedfix_best.json --family hs+near \
  --flow test=$R/flow-sf-test --output $R/speed-vwwide-hsnear > $R/speedfix.log 2>&1 || log "FAILED speed fix"
log "final entry"
OUT=$R/final DPVO=$R/dpvo-seed PI3X=$R/pi3x PI3X_W32=$R/pi3x-w32 VGGT=$R/vggt OMEGA=$R/omega512 SPEED=$R/speed-vwwide-hsnear \
  bash fermarble/jobs/final_entry.sh > $R/final.log 2>&1 || log "FAILED final entry"
cat $R/final.log
log "DONE"
