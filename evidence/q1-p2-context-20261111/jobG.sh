#!/bin/bash
# Seat job G (program revision 2, context growth). 513 tokens of chat text (the four calibration
# sequences joined), teacher-forced:
#   a. start in the 288-position functions (prefill for the first 256 tokens), grow into the
#      1024-position function when position 288 is reached
#   b. the 1024-position function from the first token
# Both are compared with the exact-arithmetic reference (ref513.f32, made beforehand with
# `anec forward`) and with each other. Then a long reply in text mode that grows twice.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; W=$R/p2; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
A=$ROOT/.build/release/anec; RUN=$ROOT/.build/release/anec-run; TK=$R/repo/tokenizer/tokenizer.json
T=$(tr ';' ',' < $R/num/calib.ids)
row() { grep -E "stateful decode|prefill \||context [0-9]+ ->|rror" | cut -c1-200; }
echo "=== a. 288 growing into 1024 $(date -u +%T)"
$RUN $W/s1/final --model $Q --tokens "$T" --generate 0 --units ane --prefill $W/s32/final --grow $W/c1024/final --dump $W/dGa --out $W/grow_a.json 2>&1 | row
echo "=== b. 1024 from the start $(date -u +%T)"
$RUN $W/c1024/final --model $Q --tokens "$T" --generate 0 --units ane --dump $W/dGb --out $W/grow_b.json 2>&1 | row
echo "a vs exact, 513: $($A compare $W/dGa/logits.ane.f32 $W/ref513.f32 --rows 513 --kl 1)"
echo "b vs exact, 513: $($A compare $W/dGb/logits.ane.f32 $W/ref513.f32 --rows 513 --kl 1)"
echo "a vs b, 513: $($A compare $W/dGa/logits.ane.f32 $W/dGb/logits.ane.f32 --rows 513 --kl 1)"
# the part past the first context only: positions 288..512
for f in dGa/logits.ane dGb/logits.ane ref513; do tail -c $((225 * 128000 * 4)) $W/$f.f32 > $W/$(basename $f).tail.f32; done
echo "a vs exact, positions 288-512: $($A compare $W/logits.ane.tail.f32 $W/ref513.tail.f32 --rows 225 --kl 1)"
rm -f $W/logits.ane.tail.f32 $W/ref513.tail.f32
rm -rf /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/dGa /Users/nayte/ane-hot/scratch/lfm3-release-20261111/p2/dGb
echo "=== long reply, growing 288 -> 1024 -> 2048 $(date -u +%T)"
$RUN $W/s1/final --model $Q --tokenizer $TK --prefill $W/s32/final --grow $W/c1024/final,$W/c2048/final --units ane \
  --prompt "Write a long, detailed essay of at least 900 words on the history of the bicycle, from the first designs to modern times, in several titled sections." \
  --max-tokens 1500 --out $W/grow_essay.json 2>&1 | fold -w 400 | cut -c1-400
echo JOB_DONE
