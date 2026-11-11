#!/bin/bash
# Seat job: three chat turns, then a 60 s x 3 sustained decode under the anvil-bench guard
# with ANE-rail power sampled by macmon (IOReport, no root) across idle / bench / idle.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; S=$R/stage; O=$R/chat
cd /Users/nayte/ane-hot/workspace/anec/src
for k in sky rope france; do
  echo "=== chat $k $(date -u +%FT%TZ)"
  $S/bin/anec-run $S/lfm3_lut4_C288 --model $S/model --tokens "$(cat $O/ids_$k)" --generate 128 --units ane --out $O/chat_$k.json | grep -E "stateful decode|seg0|host-carried|rror"
done
/opt/homebrew/bin/macmon pipe -i 1000 > $O/macmon.jsonl 2>/dev/null &
MP=$!
echo "idle_start=$(date -u +%s)"; sleep 20
echo "bench_start=$(date -u +%s)"
env ANEC_BUILD=.build-b BENCH_SKIP_CPU=1 BENCH_MIN_SECONDS=60 bash scripts/neo_bench_decode.sh $S/lfm3_lut4_C288 $S/model "$(cat $O/fixture270.ids)" 0
echo "bench_end=$(date -u +%s)"; sleep 15
echo "idle_end=$(date -u +%s)"
kill $MP 2>/dev/null
echo JOB_DONE
