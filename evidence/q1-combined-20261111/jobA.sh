#!/bin/bash
# Seat job A (combined build): compile the q1-weight checkpoint with anec into the one-token and
# the 32-token prefill functions at a 288-position context, check placement, then compare the
# ANE logits with the teacher over the 270-token fixture, with and without the prefill hand-off.
# Only aggregates are printed.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; W=$R/q1; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
FX=/Users/nayte/ane-hot/runs/lfm3-campaign-10692/claude-decode-20261108/pruned-ffn7168-b32-fixture
cd /Users/nayte/ane-hot/workspace/anec/src
A=.build-b/release/anec; ORA=$PWD/.build-b/release/anec-oracle
RUN=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec/.build/release/anec-run
F=$(cat $W/fixture270.ids)
row() { grep -E "stateful decode|prefill \||rror" | cut -c1-200; }
echo "=== fit one-token, context 288 $(date -u +%FT%TZ)"
$A fit --model $Q --seq 1 --context 288 --out $W/s1 --oracle $ORA --state-form ioDelta --start-mb 450 --palettize 4 --palette-group 8
echo "=== fit prefill 32, context 288 $(date -u +%FT%TZ)"
$A fit --model $Q --seq 32 --context 288 --out $W/s32 --oracle $ORA --state-form ioDelta --start-mb 450 --palettize 4 --palette-group 8
[ -f $W/s1/final/plan.json ] && [ -f $W/s32/final/plan.json ] || { echo "FIT_FAILED"; exit 1; }
echo "=== one-token function over 270 tokens $(date -u +%FT%TZ)"
$RUN $W/s1/final --model $Q --tokens "$F" --generate 0 --units ane --dump $W/d1 --out $W/tf_s1.json | row
echo "=== with the prefill hand-off over the same 270 tokens $(date -u +%FT%TZ)"
$RUN $W/s1/final --model $Q --tokens "$F" --generate 0 --units ane --prefill $W/s32/final --dump $W/d2 --out $W/tf_chain.json | row
head -c $((270 * 128000 * 4)) $FX/logits.f32 > $W/teacher.f32
for n in 270 64 28; do
  echo "ANE vs teacher, first $n: $($A compare $W/d1/logits.ane.f32 $W/teacher.f32 --rows $n --kl 1)"
done
echo "with prefill vs teacher, 270: $($A compare $W/d2/logits.ane.f32 $W/teacher.f32 --rows 270 --kl 1)"
echo "with vs without prefill, 270: $($A compare $W/d2/logits.ane.f32 $W/d1/logits.ane.f32 --rows 270 --kl 1)"
echo "ANE     $($A nll $W/d1/logits.ane.f32 --tokens $F)"
echo "teacher $($A nll $W/teacher.f32 --tokens $F)"
rm -f /Users/nayte/ane-hot/scratch/lfm3-release-20261111/q1/teacher.f32
rm -rf /Users/nayte/ane-hot/scratch/lfm3-release-20261111/q1/d1 /Users/nayte/ane-hot/scratch/lfm3-release-20261111/q1/d2
echo JOB_DONE
