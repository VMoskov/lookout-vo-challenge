#!/usr/bin/env bash
# flir-IR focal test: DPVO seed 0 with CLAHE at focal x0.5 and x0.75 on the 5 flir-IR dev clips,
# then on test clip_100. Start: setsid nohup bash fermarble/jobs/flirir_focal.sh > runs/flirir_focal.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
V=clip_019,clip_022,clip_024,clip_040,clip_046
echo "$(date '+%F %T') START"
for f in 0.5 0.75; do d=runs/dpvo-fps8-seed0-clahe-f$f; mkdir -p $d
  $PY fermarble/run_dpvo.py --clips $V --seed 0 --clahe --focal-scale $f --output $d >> $d/run.log 2>&1 &
done; wait
for f in 0.5 0.75; do d=runs/dpvo-test-fps8-seed0-clahe-f$f; mkdir -p $d
  $PY fermarble/run_dpvo.py --dataset lookout-vo-test-v0.1/release_manifest.json --clips clip_100 --seed 0 --clahe --focal-scale $f --output $d >> $d/run.log 2>&1 &
done; wait
for d in runs/dpvo-fps8-seed0-clahe-f0.5 runs/dpvo-fps8-seed0-clahe-f0.75 runs/dpvo-test-fps8-seed0-clahe-f0.5 runs/dpvo-test-fps8-seed0-clahe-f0.75; do
  echo "$d: $(ls $d/poses 2>/dev/null | wc -l) clips with poses; $(grep -c tracking_failed $d/run.log) failed"; done
echo "$(date '+%F %T') DONE"
