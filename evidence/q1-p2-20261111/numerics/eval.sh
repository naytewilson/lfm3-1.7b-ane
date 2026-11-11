#!/bin/bash
# eval.sh <name> <range.json|none> — compile the q1 checkpoint as the one-token function at a
# 288-position context (optionally with range lifts), then under the seat check placement and
# run the 270-token teacher fixture on the ANE. Prints aggregates only.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; N=$R/num; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
FX=/Users/nayte/ane-hot/runs/lfm3-campaign-10692/claude-decode-20261108/pruned-ffn7168-b32-fixture
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
A=$ROOT/.build/release/anec
name="${1:?name}"; range="${2:?range.json or none}"; V=$N/$name
rflag=(); [ "$range" != none ] && rflag=(--range "$range")
# any further arguments go to the compiler unchanged (for example --activation tanh)
shift 2; rflag+=("$@")
rm -rf "$V"; mkdir -p "$V"
echo "=== $name compile $(date -u +%FT%TZ) range=$range"
$A compile --model $Q --seq 1 --context 288 --palettize 4 --palette-group 8 --state-form ioDelta --max-weight-mb 450 "${rflag[@]}" --out $V/s1 2>&1 | tail -4 | cut -c1-200
[ -f $V/s1/plan.json ] || { echo COMPILE_FAILED; exit 1; }
first=$(ls -d $V/s1/*.mlpackage | head -1)
bash $ROOT/scripts/neo_seat_cmd.sh lfm3num-$name "$first" bash $N/eval_seat.sh "$name" | cut -c1-200
cat $V/seat.log 2>/dev/null | cut -c1-220
[ -f $V/d1/logits.ane.f32 ] || { echo NO_DEVICE_LOGITS; exit 1; }
head -c $((270 * 128000 * 4)) $FX/logits.f32 > $V/teacher.f32
echo "ANE vs teacher 270: $($A compare $V/d1/logits.ane.f32 $V/teacher.f32 --rows 270 --kl 1 | cut -c1-160)"
echo "position ranges: $(node $R/q1ref/rowkl.mjs $V/d1/logits.ane.f32 $V/teacher.f32 270)"
echo "ANE vs exact q1 270: $($A compare $V/d1/logits.ane.f32 $R/q1ref/logits270.q1.f32 --rows 270 --kl 1 | cut -c1-160)"
echo "ANE vs exact, position ranges: $(node $R/q1ref/rowkl.mjs $V/d1/logits.ane.f32 $R/q1ref/logits270.q1.f32 270)"
echo "ANE  $($A nll $V/d1/logits.ane.f32 --tokens "$(cat $R/q1/fixture270.ids)")"
rm -f "$V/teacher.f32"; rm -rf "$V/d1"
echo "=== $name done $(date -u +%FT%TZ)"
