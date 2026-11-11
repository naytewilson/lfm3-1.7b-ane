#!/bin/bash
# Seat job: build a 32-token-per-call (prefill) function of the LUT4 model at a 288-position
# context, check its placement, compare its logits with the one-token function and the teacher
# over the same 256 tokens, then time it under the anvil-bench guard with ANE power sampled.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; P=$R/prefill; M=$P/model-src
FX=/Users/nayte/ane-hot/runs/lfm3-campaign-10692/claude-decode-20261108/pruned-ffn7168-b32-fixture
cd /Users/nayte/ane-hot/workspace/anec/src
A=.build-b/release/anec; RUN=.build-b/release/anec-run; ORA=$PWD/.build-b/release/anec-oracle
echo "=== fit S=32 C=288 $(date -u +%FT%TZ)"
$A fit --model $M --seq 32 --context 288 --out $P/s32 --oracle $ORA --state-form ioDelta --start-mb 450 --palettize 4 --palette-group 8
F=$P/s32/final
[ -f $F/plan.json ] || { echo "FIT_FAILED: no final plan"; exit 1; }
TOK=$(cut -d, -f1-256 $P/fixture270.ids)
echo "=== prefill function, teacher-forced over 256 tokens (8 calls of 32) $(date -u +%FT%TZ)"
$RUN $F --model $M --tokens $TOK --generate 0 --units ane --dump $F/dump --out $P/tf_s32.json | grep -E "stateful decode|host-carried|seg0|rror"
echo "=== one-token function, same 256 tokens $(date -u +%FT%TZ)"
$RUN $P/lfm3_lut4_C288 --model $M --tokens $TOK --generate 0 --units ane --dump $P/s1dump --out $P/tf_s1.json | grep -E "stateful decode|host-carried|seg0|rror"
head -c $((256 * 128000 * 4)) $FX/logits.f32 > $P/teacher.f32
echo "prefill vs one-token: $($A compare $F/dump/logits.ane.f32 $P/s1dump/logits.ane.f32 --rows 256 --kl 1)"
echo "prefill vs teacher:   $($A compare $F/dump/logits.ane.f32 $P/teacher.f32 --rows 256 --kl 1)"
echo "one-token vs teacher: $($A compare $P/s1dump/logits.ane.f32 $P/teacher.f32 --rows 256 --kl 1)"
rm -rf $P/teacher.f32 $F/dump $P/s1dump
/opt/homebrew/bin/macmon pipe -i 1000 > $P/macmon.jsonl 2>/dev/null &
MP=$!
echo "idle_start=$(date -u +%s)"; sleep 20
echo "bench_start=$(date -u +%s)"
env ANEC_BUILD=.build-b BENCH_SKIP_CPU=1 BENCH_MIN_SECONDS=60 bash scripts/neo_bench_decode.sh $F $M "$TOK" 0
echo "bench_end=$(date -u +%s)"; sleep 15
echo "idle_end=$(date -u +%s)"
kill $MP 2>/dev/null
echo JOB_DONE
