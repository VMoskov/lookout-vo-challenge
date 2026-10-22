#!/usr/bin/env bash
# Foundation-model heading variants A-D on dev (official evaluator) and test (predictions only):
#   A  circular median of DPVO 3-seed median, VGGT, Pi3X (wide + flir-VIS) + broken-seed rule
#   B  Pi3X heading on every wide + flir-VIS clip, DPVO elsewhere
#   C  5-member median: 3 DPVO seeds + VGGT + Pi3X (wide + flir-VIS) + broken-seed rule
#   D  VGGT switch at 54.25 + broken-seed rule; when VGGT replaces the median, circular mean of VGGT and Pi3X
# Other camera types: DPVO 3-seed median. Outputs runs/fm-variants/{dev,test}-{A,B,C,D}.
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
O=runs/fm-variants; mkdir -p $O
build() {  # split-tag, dataset, poses, speed, vggt dir, pi3x dir
  local tag=$1 ds=$2 poses=$3 speed=$4 V=$5 P=$6
  local C="$PY fermarble/make_predictions.py --dataset $ds $( [ $tag = dev ] && echo --split train,val || echo --split test ) --poses $poses --speed $speed --speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS --overwrite"
  $C --vggt $V --switch-deg 1e9 --fm $V $P --fm-combine median3 --output $O/$tag-A > $O/$tag-A.log 2>&1 &
  $C --vggt $P --switch-modes none --flat-rule=1e9,0,-1 --flat-modes wide,flir-VIS --output $O/$tag-B > $O/$tag-B.log 2>&1 &
  $C --vggt $V --switch-deg 1e9 --fm $V $P --fm-combine members --output $O/$tag-C > $O/$tag-C.log 2>&1 &
  $C --vggt $V --fm $P --fm-combine blend --output $O/$tag-D > $O/$tag-D.log 2>&1 &
  wait
}
build dev release/release_manifest.json "runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2" runs/flow-v2 runs/vggt-dev runs/pi3x-dev
build test lookout-vo-test-v0.1/release_manifest.json "runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe" runs/flow-test runs/vggt-test runs/pi3x-test
for v in A B C D; do $PY starter/evaluate.py --dataset release/release_manifest.json --split train,val --predictions $O/dev-$v >> $O/dev-$v.log 2>&1 & done; wait
for v in A B C D; do tail -1 $O/dev-$v.log; done
