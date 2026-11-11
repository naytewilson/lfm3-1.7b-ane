#!/bin/bash
# Seat job L (program revision 2, longer contexts): fit the one-token function at 1024 and 2048
# positions (same weights, range lifts and tanh activations), check placement, compare the ANE
# logits with the teacher over the 270-token fixture, and time each under the bench guard.
# Only aggregates are printed.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; W=$R/p2; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
FX=/Users/nayte/ane-hot/runs/lfm3-campaign-10692/claude-decode-20261108/pruned-ffn7168-b32-fixture
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
A=$ROOT/.build/release/anec; ORA=$ROOT/.build/release/anec-oracle; RUN=$ROOT/.build/release/anec-run
F=$(cat $R/q1/fixture270.ids)
OPTS="--state-form ioDelta --start-mb 450 --palettize 4 --palette-group 8 --range $W/range.json --activation tanh"
head -c $((270 * 128000 * 4)) $FX/logits.f32 > $W/teacher.f32
cd $ROOT
for C in "$@"; do
  echo "=== fit one-token, context $C $(date -u +%FT%TZ)"
  $A fit --model $Q --seq 1 --context $C --out $W/c$C --oracle $ORA $OPTS 2>&1 | grep -E "^fit:|       seg|rror" | cut -c1-220
  [ -f $W/c$C/final/plan.json ] || { echo "FIT_FAILED context $C"; continue; }
  echo "=== 270 tokens on the ANE, context $C $(date -u +%FT%TZ)"
  $RUN $W/c$C/final --model $Q --tokens "$F" --generate 0 --units ane --dump $W/dL --out $W/tf_c$C.json | grep -E "stateful decode|host-carried|   seg|   host|rror" | cut -c1-200
  echo "context $C, ANE vs teacher, 270: $($A compare $W/dL/logits.ane.f32 $W/teacher.f32 --rows 270 --kl 1)"
  echo "context $C, ANE vs exact program, 270: $($A compare $W/dL/logits.ane.f32 $R/q1ref/logits270.q1.f32 --rows 270 --kl 1)"
  rm -rf "$W/dL"
  echo "=== bench, context $C $(date -u +%FT%TZ) load=$(sysctl -n vm.loadavg)"
  env BENCH_SKIP_CPU=1 BENCH_MIN_SECONDS=30 bash scripts/neo_bench_decode.sh $W/c$C/final $Q "$F" 0 2>&1 | grep -E "^ane-sustained|BENCH_EVIDENCE|REJECT|FAIL|rror" | sed -E 's/\| segments=3 \| units=cpuAndNeuralEngine \| steps=270 //' | cut -c1-200
done
rm -f /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/teacher.f32
echo JOB_DONE
