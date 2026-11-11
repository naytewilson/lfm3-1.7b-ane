#!/bin/bash
# bench.sh <rounds> <variant>... — sustained decode timing of compiled variants under the
# anvil-bench guard, with ANE power from macmon. Variants are visited in the given order, once
# per round, so each one is timed next to the others under the same machine conditions.
# Run under the seat. Before timing, wait (bounded) for a settled machine: 1-minute load below 4
# on three checks 20 s apart; the result is recorded either way.
set -uo pipefail
N=/Users/nayte/ane-hot/scratch/lfm3-release-20261111/num; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
rounds="${1:?rounds}"; shift
cd $ROOT
ok=0; for i in 1 2 3 4 5 6 7 8 9; do l=$(sysctl -n vm.loadavg | awk '{print $2}'); echo "load_check_$i=$l $(date -u +%T)"; if awk -v l="$l" 'BEGIN{exit !(l<4)}'; then ok=$((ok+1)); else ok=0; fi; [ $ok -ge 3 ] && break; sleep 20; done
echo "settled=$([ $ok -ge 3 ] && echo yes || echo no) thermal=$(pmset -g therm 2>/dev/null | grep -c 'No thermal warning')"
/opt/homebrew/bin/macmon pipe -i 1000 > $N/bench.macmon.jsonl 2>/dev/null &
MP=$!
echo "idle_start=$(date -u +%s)"; sleep 12
for r in $(seq 1 "$rounds"); do
  for v in "$@"; do
    echo "arm_start variant=$v round=$r t=$(date -u +%s) load=$(sysctl -n vm.loadavg)"
    env BENCH_SKIP_CPU=1 BENCH_MIN_SECONDS="${BENCH_MIN_SECONDS:-30}" bash scripts/neo_bench_decode.sh $N/$v/s1 $Q "$(cat $N/../q1/fixture270.ids)" 0 2>&1 | grep -E "stateful decode|BENCH_EVIDENCE|REJECT|FAIL|rror" | cut -c1-220
    echo "arm_end variant=$v round=$r t=$(date -u +%s) load=$(sysctl -n vm.loadavg)"
  done
done
sleep 10; echo "idle_end=$(date -u +%s)"
kill $MP 2>/dev/null
echo JOB_DONE
