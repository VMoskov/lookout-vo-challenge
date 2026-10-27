#!/usr/bin/env bash
# Overnight Pi3X runs for the last slot, on kwide_crop's frames (crops.json bottom crop, no inpainting):
#   kf1.5c / kf3c: the existing 1.5 / 3 fps runs, with the inpainted clips (structure-masks-auto edits) re-run without
#                  inpainting (other clips linked; those runs used crop only)
#   kf1, kf4:      1 / 4 fps, window 24
#   w16, w32:      2 fps, window 16 / 32 (current: 24)
# Test first, then dev val, then dev train (the 80 clips VGGT covers). ens_eval.sh after test and at the end;
# status_ens.py after each stage.
# Start: setsid nohup bash fermarble/jobs/ens_gpu.sh > runs/ens_gpu.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/../.."
PI=$HOME/venvs/pi3/bin/python
TEST=lookout-vo-test-v0.1/release_manifest.json
CR=runs/static-regions/crops.json
log() { echo "$(date '+%F %T') $*"; }
dev() { python3 -c "import json;s=json.load(open('runs/vggt-dev/vggt_summary.json'));print(','.join(k for k,v in s.items() if 'error' not in v and v['split']=='$1'))"; }
for sp in test dev; do
  ED=$(python3 -c "import json;print(' '.join(json.load(open('runs/structure-masks-auto/edits_$sp.json'))))")
  for f in 1.5 3; do d=runs/pi3x-$sp-kf${f}c/poses; mkdir -p $d
    for p in runs/pi3x-$sp-kf$f/poses/*.npz; do k=$(basename $p .npz)
      case " $ED " in *" $k "*) ;; *) [ -e $d/$k.npz ] || ln -s $(realpath $p) $d/$k.npz ;; esac; done
  done
done
RUNS="kf1.5c:--keyframe-fps=1.5 kf3c:--keyframe-fps=3 kf1:--keyframe-fps=1 kf4:--keyframe-fps=4 w16:--window=16 w32:--window=32"
stage() { local sp=$1 label=$2; shift 2
  for r in $RUNS; do n=${r%%:*} a=${r#*:}; d=runs/pi3x-$sp-$n; mkdir -p $d; log "$label $n"
    $PI fermarble/pi3_tool.py "$@" $a --crops $CR --output $d >> $d/run.log 2>&1 || log "FAILED $label $n"; done
  log "STAGE DONE $label"; python3 fermarble/jobs/status_ens.py > /dev/null; }
stage test test --dataset $TEST --split test
bash fermarble/jobs/ens_eval.sh >> runs/ens_eval.log 2>&1 &   # test predictions + mean5 bundle while dev runs
stage dev dev-val --clips $(dev val)
stage dev dev-train --clips $(dev train)
log "GPU DONE"
bash fermarble/jobs/ens_eval.sh >> runs/ens_eval.log 2>&1 || log "eval FAILED"
log "DONE"
