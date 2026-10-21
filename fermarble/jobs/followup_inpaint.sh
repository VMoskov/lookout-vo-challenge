#!/usr/bin/env bash
# Follow-up: (1) after vggt_focal_dpvo.sh, re-run clip_000 (its decoders were killed by mistake) for
# seeds 0-2 at the VGGT focal scale and refresh the merged links; (2) overlay-inpainting test:
# DPVO x1.0 seed 0 on 075/082 (logo + gauge inpainted) and 076 (logo only, control).
# Start: setsid nohup bash fermarble/jobs/followup_inpaint.sh > runs/followup_inpaint.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
echo "$(date '+%F %T') waiting for vggt_focal_dpvo.sh"
while pgrep -f "jobs/vggt_focal_dpv[o].sh" > /dev/null; do sleep 30; done
echo "$(date '+%F %T') re-running clip_000 at x1.492"
for s in 0 1 2; do
  $PY fermarble/run_dpvo.py --clips clip_000 --seed $s --focal-scale 1.492 --retry-failed --output runs/fvggt/s$s/clip_000 >> runs/fvggt/s$s/clip_000/run.log 2>&1 &
done; wait
for s in 0 1 2; do d=runs/dpvo-fps8-seed$s-fvggt/poses; mkdir -p $d; for c in runs/fvggt/s$s/*/poses/*.npz; do ln -sfn ../../fvggt/s$s/$(basename $(dirname $(dirname $c)))/poses/$(basename $c) $d/; done; echo "seed $s: $(ls $d | wc -l)/8"; done
echo "$(date '+%F %T') inpainting runs"
rm -rf runs/dpvo-fps8-seed0-inpaint-gauge runs/dpvo-fps8-seed0-inpaint-logo
mkdir -p runs/dpvo-fps8-seed0-inpaint-gauge runs/dpvo-fps8-seed0-inpaint-logo
$PY fermarble/run_dpvo.py --clips clip_075,clip_082 --seed 0 --inpaint-rects '[[1665,25,1855,119],[59,908,248,1051]]' --output runs/dpvo-fps8-seed0-inpaint-gauge >> runs/dpvo-fps8-seed0-inpaint-gauge/run.log 2>&1 &
$PY fermarble/run_dpvo.py --clips clip_076 --seed 0 --inpaint-rects '[[1665,25,1855,119]]' --output runs/dpvo-fps8-seed0-inpaint-logo >> runs/dpvo-fps8-seed0-inpaint-logo/run.log 2>&1 &
wait
grep -h -E "RUN_DPVO|failed" runs/dpvo-fps8-seed0-inpaint-*/run.log
echo "$(date '+%F %T') DONE"
