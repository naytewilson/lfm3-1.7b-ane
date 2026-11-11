#!/bin/bash
# Seat job C (program revision 2): chat turns through the runner's text front end, with the
# prefill function; a check of the native tokenizer against the token ids the Node tokenizer
# produced for the same prompts; and a three-turn chat read from standard input.
set -uo pipefail
R=/Users/nayte/ane-hot/scratch/lfm3-release-20261111; W=$R/p2; Q=/Users/nayte/ane-hot/models/lfm3-q1-carry-20261111
ROOT=/Users/nayte/Projects/ANE-Lab/worktrees/claude-anec-prefill-chain-20261111/anec
RUN=$ROOT/.build/release/anec-run; TK=$R/repo/tokenizer/tokenizer.json; E=$R/repo/evidence
run() { local name=$1; shift; echo "=== $name $(date -u +%T)"; "$RUN" $W/s1/final --model $Q --tokenizer $TK --prefill $W/s32/final --units ane --out $W/chat_$name.json "$@" 2>&1 | cut -c1-1200; }
run france --prompt "What is the capital of France?" --max-tokens 128
run sky --prompt "Explain in one sentence why the sky is blue." --max-tokens 128
run rope --prompt "Explain RoPE in two sentences" --max-tokens 128
# the 105-token note, given as the token ids used before; the record keeps its text
run long --tokens "$(cat $E/prefill-chain-20261111/ids_long)" --generate 64 | grep -E "stateful decode|prefill \||rror"
for k in france sky rope; do
  node -e 'const fs=require("fs");const r=JSON.parse(fs.readFileSync(process.argv[1],"utf8"));const want=fs.readFileSync(process.argv[2],"utf8").trim();console.log("tokenizer check '$k':",r.prompt.join(",")===want?"same ids as the Node tokenizer":"DIFFERENT",r.prompt.length,"tokens")' $W/chat_$k.json $E/chat-20261111/ids_$k
done
echo "=== three-turn chat $(date -u +%T)"
printf 'And what river runs through it?\nName one museum there.\n' | "$RUN" $W/s1/final --model $Q --tokenizer $TK --prefill $W/s32/final --units ane --prompt "What is the capital of France?" --chat 1 --max-tokens 96 --out $W/chat_multi.json 2>&1 | cut -c1-1200
echo JOB_DONE
