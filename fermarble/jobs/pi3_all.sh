#!/usr/bin/env bash
# Pi3X (yyfz233/Pi3X): clip_075 with injected intrinsics x1 and x2, then the 80 dev clips VGGT covers
# (flir-VIS first), then the 20 test clips. Done clips are skipped, so a rerun resumes.
# Start: setsid nohup bash fermarble/jobs/pi3_all.sh > runs/pi3_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/venvs/pi3/bin/python
mkdir -p runs/pi3x-075-k1 runs/pi3x-075-k2 runs/pi3x-dev runs/pi3x-test
echo "$(date '+%F %T') START 075 injected intrinsics"
$PY fermarble/pi3_tool.py --clips clip_075 --inject-focal 1.0 --output runs/pi3x-075-k1 >> runs/pi3x-075-k1/run.log 2>&1
$PY fermarble/pi3_tool.py --clips clip_075 --inject-focal 2.0 --output runs/pi3x-075-k2 >> runs/pi3x-075-k2/run.log 2>&1
echo "$(date '+%F %T') START dev flir-VIS"
$PY fermarble/pi3_tool.py --clips clip_000,clip_001,clip_075,clip_076,clip_082,clip_084,clip_088,clip_090 --output runs/pi3x-dev >> runs/pi3x-dev/run.log 2>&1
echo "$(date '+%F %T') START dev rest"
$PY fermarble/pi3_tool.py --clips clip_002,clip_003,clip_004,clip_005,clip_006,clip_007,clip_008,clip_009,clip_010,clip_011,clip_012,clip_013,clip_014,clip_016,clip_017,clip_018,clip_019,clip_020,clip_021,clip_022,clip_023,clip_024,clip_025,clip_026,clip_027,clip_029,clip_030,clip_031,clip_033,clip_034,clip_035,clip_036,clip_037,clip_038,clip_040,clip_041,clip_043,clip_044,clip_046,clip_047,clip_049,clip_050,clip_051,clip_052,clip_053,clip_054,clip_058,clip_059,clip_060,clip_061,clip_062,clip_063,clip_064,clip_065,clip_066,clip_067,clip_068,clip_070,clip_071,clip_072,clip_073,clip_074,clip_077,clip_078,clip_079,clip_080,clip_081,clip_083,clip_085,clip_086,clip_087,clip_091 --output runs/pi3x-dev >> runs/pi3x-dev/run.log 2>&1
echo "$(date '+%F %T') START test"
$PY fermarble/pi3_tool.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --output runs/pi3x-test >> runs/pi3x-test/run.log 2>&1
echo "$(date '+%F %T') DONE"
