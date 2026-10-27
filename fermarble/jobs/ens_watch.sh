#!/usr/bin/env bash
# Deadline guard for the overnight ensembles: at 08:15, if ens_gpu.sh has not finished, evaluate what is complete
# (ens_eval.sh) so runs/status_ens.md has a recommendation by 09:00; the GPU runs keep going.
# Start: setsid nohup bash fermarble/jobs/ens_watch.sh > runs/ens_watch.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
sleep $(( $(date -d 'today 08:15' +%s) - $(date +%s) ))
grep -q "GPU DONE" runs/ens_gpu.log || bash fermarble/jobs/ens_eval.sh >> runs/ens_eval.log 2>&1
echo "$(date '+%F %T') watch done"
