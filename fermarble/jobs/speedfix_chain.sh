#!/usr/bin/env bash
# Waits for speedfix_flow.sh, then runs speedfix_eval.sh.
# Start: setsid nohup bash fermarble/jobs/speedfix_chain.sh > runs/speedfix_chain.log 2>&1 < /dev/null &
cd "$(dirname "$0")/../.."
until grep -q DONE runs/speedfix_flow.log; do sleep 60; done
bash fermarble/jobs/speedfix_eval.sh > runs/speedfix_eval.log 2>&1
echo "$(date '+%F %T') chain done"
