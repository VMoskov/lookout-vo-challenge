#!/usr/bin/env bash
# VGGT on all 67 wide train+val clips (done clips are skipped), then on all 20 test clips.
# Start: setsid nohup bash fermarble/jobs/vggt_all.sh > runs/vggt_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PY=$HOME/venvs/vggt/bin/python
echo "$(date '+%F %T') START wide dev"
$PY fermarble/vggt_tool.py --clips clip_002,clip_003,clip_004,clip_005,clip_006,clip_007,clip_008,clip_009,clip_010,clip_011,clip_012,clip_013,clip_014,clip_016,clip_017,clip_018,clip_020,clip_021,clip_023,clip_025,clip_026,clip_027,clip_029,clip_030,clip_031,clip_033,clip_034,clip_035,clip_036,clip_037,clip_038,clip_041,clip_043,clip_044,clip_047,clip_049,clip_050,clip_051,clip_052,clip_053,clip_054,clip_058,clip_059,clip_060,clip_061,clip_062,clip_063,clip_064,clip_065,clip_066,clip_067,clip_068,clip_070,clip_071,clip_072,clip_073,clip_074,clip_077,clip_078,clip_079,clip_080,clip_081,clip_083,clip_085,clip_086,clip_087,clip_091 --output runs/vggt-dev >> runs/vggt-dev/run.log 2>&1
echo "$(date '+%F %T') START test"
mkdir -p runs/vggt-test
$PY fermarble/vggt_tool.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --output runs/vggt-test >> runs/vggt-test/run.log 2>&1
echo "$(date '+%F %T') DONE"
