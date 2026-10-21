#!/usr/bin/env bash
# Re-run VGGT on the test clips that hit CUDA OOM (clips without poses), after the DPVO follow-up job is done.
set -u
cd "$(dirname "$0")/../.."
echo "$(date '+%F %T') waiting for followup_inpaint.sh"
while pgrep -f "jobs/followup_inpain[t].sh|jobs/vggt_focal_dpv[o].sh" > /dev/null; do sleep 30; done
python3 - <<'PY'
import json; p='runs/vggt-test/vggt_summary.json'; s=json.load(open(p))
json.dump({k: v for k, v in s.items() if 'error' not in v}, open(p, 'w'), indent=2)
PY
echo "$(date '+%F %T') START"
$HOME/venvs/vggt/bin/python fermarble/vggt_tool.py --dataset lookout-vo-test-v0.1/release_manifest.json --split test --output runs/vggt-test >> runs/vggt-test/run.log 2>&1
python3 -c "
import json; s=json.load(open('runs/vggt-test/vggt_summary.json')); print(len(s), 'test clips; errors:', [k for k,v in s.items() if 'error' in v])"
echo "$(date '+%F %T') DONE"
