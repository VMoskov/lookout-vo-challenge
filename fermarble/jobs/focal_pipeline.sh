#!/usr/bin/env bash
# Detached flir-VIS focal-length pipeline (start with setsid nohup, see bottom of this header).
#   A  step 1: 8 flir-VIS train clips at focal x2.0, seeds 1 and 2 (seed 0 exists)
#      step 2: same clips at x1.0/1.5/2.0/2.5, seed 0, with DPVO residual/confidence (--dpvo-diag)
#      (run_dpvo.py resumes: finished clips are skipped; one --retry-failed pass afterwards)
#   B  report steps 1+2 -> runs/focal-report/focal_report.txt
#   C  step 3: 11 flir-VIS test clips at x2.0, seeds 0/1/2 (+ one retry pass)
#   D  flow speed for those 11 clips with --focal-scale 2.0
#   E  med3 heading (+CLAHE flir-IR) with flow v3a on wide + flir-VIS, x2.0 for flir-VIS
#      -> runs/submission_test_flow_focal.json, validated with starter/pack_submission.py. Not uploaded.
# Start:  setsid nohup bash fermarble/jobs/focal_pipeline.sh > runs/focal_pipeline.log 2>&1 < /dev/null &
# Progress: tail -n 30 runs/focal_pipeline.log ; cat runs/focal_pipeline.status
set -u
cd "$(dirname "$0")/../.."                      # lookout-vo-challenge
PY=$HOME/micromamba/envs/dpvo/bin/python
DEV=release/release_manifest.json
TEST=lookout-vo-test-v0.1/release_manifest.json
VDEV=clip_000,clip_001,clip_075,clip_076,clip_082,clip_084,clip_088,clip_090
VTEST=clip_093,clip_094,clip_095,clip_096,clip_098,clip_099,clip_104,clip_107,clip_108,clip_109,clip_111
STATUS=runs/focal_pipeline.status
MAXGPU=4                                          # concurrent DPVO processes (16 GB GPU)
log() { echo "$(date '+%F %T') $*"; echo "$(date '+%F %T') $*" > "$STATUS"; }
pool() { xargs -P "$MAXGPU" -I{} bash -c '{}'; }  # run the command lines on stdin, MAXGPU at a time

log "START pid $$ (session $(ps -o sid= -p $$ | tr -d ' '))"

# ---------------------------------------------------------------------------- A
log "A: steps 1+2 DPVO (resumable)"
A_CMDS=()
for s in 1 2; do A_CMDS+=("$PY fermarble/run_dpvo.py --clips $VDEV --seed $s --focal-scale 2.0 --output runs/dpvo-fps8-seed$s-f2 >> runs/dpvo-fps8-seed$s-f2/run.log 2>&1"); done
for f in 1.0 1.5 2.0 2.5; do A_CMDS+=("$PY fermarble/run_dpvo.py --clips $VDEV --seed 0 --focal-scale $f --dpvo-diag --output runs/diag-f$f >> runs/diag-f$f/run.log 2>&1"); done
for s in 1 2; do mkdir -p runs/dpvo-fps8-seed$s-f2; done
for f in 1.0 1.5 2.0 2.5; do mkdir -p runs/diag-f$f; done
printf '%s\n' "${A_CMDS[@]}" | pool
printf '%s --retry-failed\n' "${A_CMDS[@]}" | sed 's/ >> \(.*\) 2>&1 --retry-failed$/ --retry-failed >> \1 2>\&1/' | pool
for d in runs/dpvo-fps8-seed1-f2 runs/dpvo-fps8-seed2-f2 runs/diag-f1.0 runs/diag-f1.5 runs/diag-f2.0 runs/diag-f2.5; do
  echo "  $d: $(ls $d/poses 2>/dev/null | wc -l)/8 clips with poses"; done

# ---------------------------------------------------------------------------- B
log "B: focal report (steps 1+2)"
$PY fermarble/oracle.py --clips $VDEV --output runs/focal-report \
  --focal-base runs/dpvo-fps8-seed0 runs/dpvo-fps8-seed1 runs/dpvo-fps8-seed2 \
  --focal-new runs/dpvo-fps8-seed0-f2 runs/dpvo-fps8-seed1-f2 runs/dpvo-fps8-seed2-f2 \
  --focal-diag 1.0=runs/diag-f1.0 1.5=runs/diag-f1.5 2.0=runs/diag-f2.0 2.5=runs/diag-f2.5 || log "B FAILED (continuing)"

# ---------------------------------------------------------------------------- C
log "C: step 3 DPVO on 11 flir-VIS test clips at x2.0, seeds 0 1 2"
C_CMDS=()
for s in 0 1 2; do mkdir -p runs/dpvo-test-fps8-seed$s-f2
  C_CMDS+=("$PY fermarble/run_dpvo.py --dataset $TEST --clips $VTEST --seed $s --focal-scale 2.0 --output runs/dpvo-test-fps8-seed$s-f2 >> runs/dpvo-test-fps8-seed$s-f2/run.log 2>&1"); done
printf '%s\n' "${C_CMDS[@]}" | pool
printf '%s --retry-failed\n' "${C_CMDS[@]}" | sed 's/ >> \(.*\) 2>&1 --retry-failed$/ --retry-failed >> \1 2>\&1/' | pool
for s in 0 1 2; do echo "  runs/dpvo-test-fps8-seed$s-f2: $(ls runs/dpvo-test-fps8-seed$s-f2/poses 2>/dev/null | wc -l)/11 clips with poses"; done

# ---------------------------------------------------------------------------- D
log "D: flow speed for the 11 flir-VIS test clips at x2.0"
nice $PY fermarble/flow_speed.py --dataset $TEST --clips $VTEST --modes flir-VIS --focal-scale 2.0 \
  --poses runs/dpvo-test-fps8-seed0-f2 runs/dpvo-test-fps8-seed1-f2 runs/dpvo-test-fps8-seed2-f2 \
  --output runs/flow-test-f2 --workers 11 || log "D FAILED (continuing)"

# ---------------------------------------------------------------------------- E
log "E: build runs/submission_test_flow_focal.json"
for s in 0 1 2; do
  d=runs/merged-test-seed$s-focal/poses; rm -rf "runs/merged-test-seed$s-focal"; mkdir -p "$d"
  for f in runs/merged-test-seed$s-clahe/poses/*.npz; do ln -s "../../merged-test-seed$s-clahe/poses/$(basename $f)" "$d/"; done
  for c in ${VTEST//,/ }; do
    if [ -f runs/dpvo-test-fps8-seed$s-f2/poses/$c.npz ]; then ln -sfn "../../dpvo-test-fps8-seed$s-f2/poses/$c.npz" "$d/$c.npz"
    else echo "  WARNING seed $s $c: no x2.0 poses, keeping x1.0"; fi
  done
done
sp=runs/flow-test-focal/speed; rm -rf runs/flow-test-focal; mkdir -p "$sp"
for f in runs/flow-test/speed/*.npz; do ln -s "../../flow-test/speed/$(basename $f)" "$sp/"; done
for c in ${VTEST//,/ }; do
  if [ -f runs/flow-test-f2/speed/$c.npz ]; then ln -sfn "../../flow-test-f2/speed/$c.npz" "$sp/$c.npz"
  else rm -f "$sp/$c.npz"; echo "  WARNING $c: no x2.0 flow speed -> constant speed"; fi
done
rm -rf runs/heading-test-med3-flow-focal
$PY fermarble/make_predictions.py --dataset $TEST --split test \
  --poses runs/merged-test-seed0-focal runs/merged-test-seed1-focal runs/merged-test-seed2-focal \
  --speed runs/flow-test-focal --speed-params runs/speed-tune3/params_v3a_no_hjump.json --speed-modes wide,flir-VIS \
  --output runs/heading-test-med3-flow-focal
OUT=runs/submission_test_flow_focal.json
if [ -e "$OUT" ]; then OUT=runs/submission_test_flow_focal_$(date +%H%M).json; echo "  $OUT used (the default name already exists)"; fi
$PY starter/pack_submission.py --dataset $TEST --split test --predictions runs/heading-test-med3-flow-focal --output "$OUT" \
  && log "E: PACKED $OUT" || log "E: PACK FAILED"
$PY - "$OUT" <<'EOF'
import json, sys, numpy as np
sys.path.insert(0, 'starter')
from public_dataset import load_manifest, load_timestamps
out = sys.argv[1]
m = load_manifest('lookout-vo-test-v0.1/release_manifest.json')
b = json.load(open(out)) if out and __import__('os').path.exists(out) else {}
pm = json.load(open('runs/heading-test-med3-flow-focal/prediction_manifest.json'))['clips']
vis = set('clip_093,clip_094,clip_095,clip_096,clip_098,clip_099,clip_104,clip_107,clip_108,clip_109,clip_111'.split(','))
f = lambda x: '-' if x is None else f'{x:.2f}'
print(f'bundle {out}: {len(b)} clips, all test clips present: {sorted(b) == sorted(c["key"] for c in m["clips"])}')
for c in m['clips']:
    k = c['key']; i = pm[k]
    used = 'FLOW' if i['method'] == 'dpvo_heading_flow_speed' else 'constant'
    print(f"{k} {c['camera_mode']:<9} focal x{'2.0' if k in vis else '1.0'}  speed {used:<8} ({i.get('speed_fallback', 'mode off')})"
          f"  runs {i.get('n_runs_used')}/{i.get('n_runs')}  reliable {f(i.get('reliable_frac'))} CV {f(i.get('flow_cv'))}"
          f"  frames {len(b[k]['times']) if k in b else 0}/{len(load_timestamps('lookout-vo-test-v0.1', c))}")
EOF
log "DONE (not uploaded)"
