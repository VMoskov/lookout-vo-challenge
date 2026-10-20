#!/usr/bin/env bash
# Option A: DPVO with CENTROID_SEL_STRAT=GRADIENT_BIAS on the 67 wide train+val clips (seed 0, x1.0).
# Two halves in separate output dirs (each run_dpvo.py owns its run_manifest.json), then the poses are
# merged by symlink into runs/dpvo-fps8-seed0-gb/poses.
# Start: setsid nohup bash fermarble/jobs/gradient_bias.sh > runs/gradient_bias.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/micromamba/envs/dpvo/bin/python
echo "$(date '+%F %T') START"
for h in a b; do mkdir -p runs/dpvo-fps8-seed0-gb-$h; done
$PY fermarble/run_dpvo.py --clips clip_002,clip_004,clip_006,clip_008,clip_010,clip_012,clip_014,clip_017,clip_020,clip_023,clip_026,clip_029,clip_031,clip_034,clip_036,clip_038,clip_043,clip_047,clip_050,clip_052,clip_054,clip_059,clip_061,clip_063,clip_065,clip_067,clip_070,clip_072,clip_074,clip_078,clip_080,clip_083,clip_086,clip_091 --seed 0 --patch-strategy GRADIENT_BIAS --output runs/dpvo-fps8-seed0-gb-a >> runs/dpvo-fps8-seed0-gb-a/run.log 2>&1 &
$PY fermarble/run_dpvo.py --clips clip_003,clip_005,clip_007,clip_009,clip_011,clip_013,clip_016,clip_018,clip_021,clip_025,clip_027,clip_030,clip_033,clip_035,clip_037,clip_041,clip_044,clip_049,clip_051,clip_053,clip_058,clip_060,clip_062,clip_064,clip_066,clip_068,clip_071,clip_073,clip_077,clip_079,clip_081,clip_085,clip_087 --seed 0 --patch-strategy GRADIENT_BIAS --output runs/dpvo-fps8-seed0-gb-b >> runs/dpvo-fps8-seed0-gb-b/run.log 2>&1 &
wait
$PY fermarble/run_dpvo.py --clips clip_002,clip_004,clip_006,clip_008,clip_010,clip_012,clip_014,clip_017,clip_020,clip_023,clip_026,clip_029,clip_031,clip_034,clip_036,clip_038,clip_043,clip_047,clip_050,clip_052,clip_054,clip_059,clip_061,clip_063,clip_065,clip_067,clip_070,clip_072,clip_074,clip_078,clip_080,clip_083,clip_086,clip_091 --seed 0 --patch-strategy GRADIENT_BIAS --retry-failed --output runs/dpvo-fps8-seed0-gb-a >> runs/dpvo-fps8-seed0-gb-a/run.log 2>&1
$PY fermarble/run_dpvo.py --clips clip_003,clip_005,clip_007,clip_009,clip_011,clip_013,clip_016,clip_018,clip_021,clip_025,clip_027,clip_030,clip_033,clip_035,clip_037,clip_041,clip_044,clip_049,clip_051,clip_053,clip_058,clip_060,clip_062,clip_064,clip_066,clip_068,clip_071,clip_073,clip_077,clip_079,clip_081,clip_085,clip_087 --seed 0 --patch-strategy GRADIENT_BIAS --retry-failed --output runs/dpvo-fps8-seed0-gb-b >> runs/dpvo-fps8-seed0-gb-b/run.log 2>&1
mkdir -p runs/dpvo-fps8-seed0-gb/poses
for h in a b; do for f in runs/dpvo-fps8-seed0-gb-$h/poses/*.npz; do ln -sfn ../../dpvo-fps8-seed0-gb-$h/poses/$(basename $f) runs/dpvo-fps8-seed0-gb/poses/; done; done
echo "$(date '+%F %T') DONE: $(ls runs/dpvo-fps8-seed0-gb/poses | wc -l)/67 clips with poses"
