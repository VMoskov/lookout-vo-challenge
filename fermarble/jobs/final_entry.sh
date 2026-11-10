#!/usr/bin/env bash
# Final entry (leaderboard "med3-flow-pi3x-omega-ir", 28 Oct 2026) built from existing model runs, with no clip-ID
# special cases: own-bow crops by select_crops.py (3-35 % for every camera type), the CLAHE DPVO run chosen for
# thermal camera modes (merge_runs.py --rule thermal), cropped Pi3X chosen for the clips in the crops file.
#   wide:     weighted circular mean of Pi3X w24 (cropped where listed), Pi3X w32 and Omega, 1/3 each, yaw x 0.95
#   flir-VIS: Omega heading; flir-IR: agreement rule forced on with Omega as both members
#   speed:    speed fix hsnear (runs/speedfix/speed-vwwide-hsnear, params fitted on train) on wide and flir-VIS
# Model runs are taken as they are; their recorded crop_bottom is checked against the crops file. Writes OUT/ and
# OUT/submission_test.json, prints its sha256 against the entry as submitted. Never uploads.
#   OUT=runs/final-entry-v2 bash fermarble/jobs/final_entry.sh
set -eu
cd "$(dirname "$0")/../.."
source fermarble/jobs/runtime.sh
PY=$FERMARBLE_DPVO_PYTHON
fermarble_require_python "$PY" dpvo
TEST=lookout-vo-test-v0.1/release_manifest.json
OUT=${OUT:-runs/final-entry-v2}
CROPS=${CROPS:-runs/static-regions/crops_v2.json}
DPVO=${DPVO:-runs/dpvo-test-fps8-seed}          # + 0/1/2, and 0/1/2-clahe for thermal clips
PI3X=${PI3X:-runs/pi3x-test}                    # uncropped w24; cropped: $PI3X-crop
PI3X_W32=${PI3X_W32:-runs/pi3x-test-w32}
VGGT=${VGGT:-runs/vggt-test}
OMEGA=${OMEGA:-runs/omega512-test}
S=${SPEED:-runs/speedfix/speed-vwwide-hsnear}
REF_SHA=f5e361160dd1dedad0b22de25e7e0175661bc129a2ffcb00724547e4ac4e9a27
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
for s in 0 1 2; do
  $PY fermarble/jobs/merge_runs.py --dataset $TEST --rule thermal --base $DPVO$s --alt $DPVO$s-clahe --output $OUT/dpvo-seed$s
done
$PY fermarble/jobs/merge_runs.py --dataset $TEST --rule crops --crops $CROPS --base $PI3X --alt $PI3X-crop --output $OUT/pi3x-w24
$PY - $TEST $CROPS $PI3X_W32 $OMEGA <<'EOF'
import json, sys, numpy as np
test, crops, runs = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3:]
for c in json.load(open(test))['clips']:
    for r in runs:
        z = np.load(f"{r}/poses/{c['key']}.npz")
        got = float(z['crop_bottom']) if 'crop_bottom' in z.files else 0.0
        assert got == float(crops.get(c['key'], 0.0)), (r, c['key'], got, crops.get(c['key']))
print('crop_bottom matches the crops file in', ', '.join(runs))
EOF
B="--speed-modes wide,flir-VIS --overwrite --vggt-label pi3x --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --agree-rule 1e9,1e9 --yaw-scale wide=0.95"
TS="--dataset $TEST --split test --poses $OUT/dpvo-seed0 $OUT/dpvo-seed1 $OUT/dpvo-seed2 --agree-a $OMEGA --agree-b $OMEGA"
W="--vggt $OUT/pi3x-w24 $PI3X_W32 $VGGT $OMEGA --vggt-combine mean --vggt-weights 0.5,0.5,0,0 wide=0.3333,0.3333,0,0.3333 flir-VIS=0,0,0,1"
nice $PY fermarble/make_predictions.py $TS $B $W --speed $S-test --speed-params $S.params.json --output $OUT/predictions-test > $OUT/predictions-test.log 2>&1
$PY starter/pack_submission.py --dataset $TEST --split test --predictions $OUT/predictions-test --output $OUT/submission_test.json > $OUT/pack.log 2>&1
sha=$(sha256sum $OUT/submission_test.json | cut -d' ' -f1)
echo "sha256 $sha"
[ "$sha" = "$REF_SHA" ] && echo "identical to the entry as submitted" || echo "DIFFERS from the entry as submitted ($REF_SHA)"
