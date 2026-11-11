#!/bin/bash
# Seat job A (program revision 2): the q1-weight checkpoint compiled with range lifts and the
# tanh activation form into the one-token and the 32-token prefill functions at a 288-position
# context. Checks placement (the fit asks the hardware oracle), then compares the ANE logits
# with the teacher over the 270-token fixture, with and without the prefill hand-off.
# Only aggregates are printed.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; W=$R/p2; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
FX=/Users/nayte/ane-hot/runs/lfm3-campaign-10692/claude-decode-20261108/pruned-ffn7168-b32-fixture
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
A=$ROOT/.build/release/anec; ORA=$ROOT/.build/release/anec-oracle; RUN=$ROOT/.build/release/anec-run
F=$(cat $R/q1/fixture270.ids)
OPTS="--state-form ioDelta --start-mb 450 --palettize 4 --palette-group 8 --range $W/range.json --activation tanh"
row() { grep -E "stateful decode|prefill \||rror" | cut -c1-200; }
echo "=== fit one-token, context 288 $(date -u +%FT%TZ)"
$A fit --model $Q --seq 1 --context 288 --out $W/s1 --oracle $ORA $OPTS
echo "=== fit prefill 32, context 288 $(date -u +%FT%TZ)"
$A fit --model $Q --seq 32 --context 288 --out $W/s32 --oracle $ORA $OPTS
[ -f $W/s1/final/plan.json ] && [ -f $W/s32/final/plan.json ] || { echo "FIT_FAILED"; exit 1; }
echo "=== one-token function over 270 tokens $(date -u +%FT%TZ)"
$RUN $W/s1/final --model $Q --tokens "$F" --generate 0 --units ane --dump $W/d1 --out $W/tf_s1.json | row
echo "=== with the prefill hand-off over the same 270 tokens $(date -u +%FT%TZ)"
$RUN $W/s1/final --model $Q --tokens "$F" --generate 0 --units ane --prefill $W/s32/final --dump $W/d2 --out $W/tf_chain.json | row
head -c $((270 * 128000 * 4)) $FX/logits.f32 > $W/teacher.f32
echo "ANE vs teacher, 270: $($A compare $W/d1/logits.ane.f32 $W/teacher.f32 --rows 270 --kl 1)"
head -c $((64 * 128000 * 4)) $W/d1/logits.ane.f32 > $W/a64.f32; head -c $((64 * 128000 * 4)) $W/teacher.f32 > $W/t64.f32
echo "ANE vs teacher, first 64: $($A compare $W/a64.f32 $W/t64.f32 --rows 64 --kl 1)"
echo "ANE vs the same program in exact arithmetic, 270: $($A compare $W/d1/logits.ane.f32 $R/q1ref/logits270.q1.f32 --rows 270 --kl 1)"
echo "with prefill vs teacher, 270: $($A compare $W/d2/logits.ane.f32 $W/teacher.f32 --rows 270 --kl 1)"
echo "with vs without prefill, 270: $($A compare $W/d2/logits.ane.f32 $W/d1/logits.ane.f32 --rows 270 --kl 1)"
echo "ANE     $($A nll $W/d1/logits.ane.f32 --tokens $F)"
echo "teacher $($A nll $W/teacher.f32 --tokens $F)"
rm -f /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/teacher.f32 /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/a64.f32 /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/t64.f32
rm -rf /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/d1 /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/d2
echo JOB_DONE
