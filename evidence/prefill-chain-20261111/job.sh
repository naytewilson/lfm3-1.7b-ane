#!/bin/bash
# Seat job: the --prefill hand-off in anec-run against the one-token path.
#   1. regression: the new runner without --prefill against the published runner, 270 tokens
#   2. parity: with --prefill against without, same 270 tokens, logits dumped
#   3. a 105-token chat prompt, 64 new tokens, three runs each way
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; C=$R/chain
A=/Users/nayte/ane-hot/workspace/anec/src/.build-b/release/anec
RUN=$C/anec-run-new; OLD=$C/lfm3-ane-runner/anec-run; M=$C/lfm3-ane-runner/model; D=$C/lfm3_lut4_C288; PF=$C/lfm3_lut4_C288_prefill32
F=$(cat $C/fixture270.ids); L=$(tr -d '\n' < $C/ids_long)
row() { grep -E "stateful decode|prefill|rror" | cut -c1-220; }
echo "=== warm both plans $(date -u +%FT%TZ)"
$RUN $D --model $M --tokens "$(echo $F | cut -d, -f1-33)" --generate 0 --units ane --prefill $PF --out $C/warm.json | row
echo "=== published runner, 270 tokens"
$OLD $D --model $M --tokens "$F" --generate 0 --units ane --out $C/tf_old.json | row
echo "=== new runner without --prefill, 270 tokens"
$RUN $D --model $M --tokens "$F" --generate 0 --units ane --dump $C/d_plain --out $C/tf_plain.json | row
echo "=== new runner with --prefill, 270 tokens"
$RUN $D --model $M --tokens "$F" --generate 0 --units ane --prefill $PF --dump $C/d_chain --out $C/tf_chain.json | row
echo "with vs without prefill: $($A compare $C/d_chain/logits.ane.f32 $C/d_plain/logits.ane.f32 --rows 270 --kl 1)"
rm -rf $C/d_plain $C/d_chain
for i in 1 2 3; do
  echo "=== chat 105-token prompt, run $i, without --prefill"
  $RUN $D --model $M --tokens "$L" --generate 64 --units ane --out $C/chat_plain_$i.json | row
  echo "=== chat 105-token prompt, run $i, with --prefill"
  $RUN $D --model $M --tokens "$L" --generate 64 --units ane --prefill $PF --out $C/chat_chain_$i.json | row
done
echo JOB_DONE
