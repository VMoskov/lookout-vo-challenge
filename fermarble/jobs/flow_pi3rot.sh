#!/usr/bin/env bash
# Flow speed with Pi3X rotations (derotation, up) instead of DPVO's, wide + flir-VIS, dev then test.
# Start: setsid nohup bash fermarble/jobs/flow_pi3rot.sh > runs/flow_pi3rot.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
echo "$(date '+%F %T') dev"
nice $PY fermarble/flow_speed.py --split train,val --modes wide,flir-VIS --poses runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 \
  --rot-poses runs/pi3x-dev --output runs/flow-v2-pi3rot --workers 8
echo "$(date '+%F %T') test"
nice $PY fermarble/flow_speed.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --modes wide,flir-VIS \
  --poses runs/merged-test-seed0-clahe runs/merged-test-seed1-clahe runs/merged-test-seed2-clahe --rot-poses runs/pi3x-test --output runs/flow-test-pi3rot --workers 8
echo "$(date '+%F %T') DONE"
