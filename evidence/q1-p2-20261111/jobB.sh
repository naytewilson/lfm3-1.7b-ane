#!/bin/bash
# Seat job B (program revision 2): 60-second decode runs under the anvil-bench guard with ANE power.
# Before timing, wait (bounded) for a settled machine: 1-minute load below 4 on three checks 20 s apart.
set -uo pipefail
W=/Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
cd /Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
ok=0; for i in 1 2 3 4 5 6 7 8 9 10 11 12; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "load_check_$i=$l $(date -u +%T)"; if awk -v l="$l" 'BEGIN{exit !(l<4)}'; then ok=$((ok+1)); else ok=0; fi; [ $ok -ge 3 ] && break; sleep 20; done
echo "settled=$([ $ok -ge 3 ] && echo yes || echo no) thermal=$(pmset -g therm 2>/dev/null | grep -c 'No thermal warning')"
/opt/homebrew/bin/macmon pipe -i 1000 > $W/macmon.jsonl 2>/dev/null &
MP=$!
echo "idle_start=$(date -u +%s)"; sleep 15
echo "bench_start=$(date -u +%s) load=$(sysctl -n vm.loadavg)"
env BENCH_SKIP_CPU=1 BENCH_MIN_SECONDS=60 bash scripts/neo_bench_decode.sh $W/s1/final $Q "$(cat $W/../q1/fixture270.ids)" 0
echo "bench_end=$(date -u +%s) load=$(sysctl -n vm.loadavg)"; sleep 12
echo "idle_end=$(date -u +%s)"
kill $MP 2>/dev/null
echo JOB_DONE
