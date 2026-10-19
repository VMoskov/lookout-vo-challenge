#!/usr/bin/env bash
# Residual seed-to-seed spread: run_dpvo --dpvo-diag at x1.0 and x2.0, seeds 1 and 2, 8 flir-VIS train clips.
# Start: setsid nohup bash fermarble/jobs/diag_seeds.sh > runs/diag_seeds.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
V=clip_000,clip_001,clip_075,clip_076,clip_082,clip_084,clip_088,clip_090
echo "$(date '+%F %T') START"
for s in 1 2; do for f in 1.0 2.0; do
  mkdir -p runs/diag-s$s-f$f
  $PY fermarble/run_dpvo.py --clips $V --seed $s --focal-scale $f --dpvo-diag --output runs/diag-s$s-f$f >> runs/diag-s$s-f$f/run.log 2>&1 &
done; done
wait
for s in 1 2; do for f in 1.0 2.0; do echo "runs/diag-s$s-f$f: $(ls runs/diag-s$s-f$f/poses | wc -l)/8"; done; done
echo "$(date '+%F %T') DONE"
